"""Separate Mechanical APDL transient solid solve driven by CFD temperatures."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil

import numpy as np
import yaml

from .thermal_loading import (
    ThermalAnalysisError, ThermalLoad, auto_mesh_size, integrate_boundary,
    prepare_load, read_boundary, validate_settings, positive,
)
from .trade_study import thermal_screening_metrics

MODEL_VERSION = "3.0-temperature-convection"
ENDPOINT_MODEL_VERSION = "4.0-endpoint-convection"
PRESCRIBED_MODEL_VERSION = "5.0-prescribed-temperature"
AEROTHERMAL_MODEL_VERSION = "6.0-aerodynamic-surface-load"
STEFAN_BOLTZMANN = 5.670374419e-8


def _progress(message):
    print(f"[thermal] {message}", flush=True)


def load_yaml(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ThermalAnalysisError(f"Configuration must be a YAML file: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ThermalAnalysisError(f"Configuration must contain a mapping: {path}")
    return cfg


def config_paths(cfg, ta_cfg, thermal_config_path, run_dir=None):
    work = Path(cfg["project"]["work_dir"]).resolve()
    if run_dir is not None and Path(run_dir).resolve() != work:
        raise ThermalAnalysisError(f"--run-dir disagrees with project.work_dir in the selected case config: {work}")
    boundary = ta_cfg.get("boundary_csv")
    if boundary:
        boundary = Path(boundary)
        if not boundary.is_absolute():
            boundary = Path(thermal_config_path).resolve().parent / boundary
    else:
        directory = Path(cfg.get("thermal_export", {}).get("output_directory", "thermal_bc"))
        if directory.is_absolute() or ".." in directory.parts:
            raise ThermalAnalysisError("thermal_export.output_directory must be inside the selected case directory.")
        boundary = work / directory / "body_wall_faces.csv"
    return work, boundary.resolve(), work / "thermal_analysis"


def temperature_stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ThermalAnalysisError("MAPDL returned invalid absolute temperatures; results were not clamped or discarded.")
    return {"minimum_K": float(values.min()), "mean_K": float(values.mean()), "maximum_K": float(values.max())}


def select_snapshots(nsets, max_frames):
    return set(int(i) for i in np.linspace(1, nsets, min(nsets, max(2, int(max_frames))), dtype=int)) if nsets else set()


def map_wall_nodes(xy, node_xy, mesh_size):
    indices, distances = [], []
    for point in xy:
        distances2 = np.sum((node_xy - point) ** 2, axis=1)
        index = int(np.argmin(distances2))
        indices.append(index)
        distances.append(float(np.sqrt(distances2[index])))
    if max(distances) > max(1e-9, float(mesh_size) * 1e-3):
        raise ThermalAnalysisError("A CFD boundary keypoint has no matching thermal mesh node; check the reconstructed mesh before mapping loads/results.")
    return np.asarray(indices, dtype=int), np.asarray(distances)


def wall_rows(time, load, wall_temperature, nnum, indices, distances, settings):
    if load.mode == "prescribed_temperature":
        # Prescribed temperature determines neither convection h nor the heat
        # needed to enforce the constraint. A reaction extraction would be
        # required for a heat-load report; do not invent zero flux here.
        return [
            {
                "time_s": float(time), "surface_order_index": index,
                "x_m": float(point[0]), "y_m": float(point[1]),
                "prescribed_surface_temperature_K": float(load.surface_temperature[index]),
                "boundary_load_state": "initial_before_application" if time == 0 else "prescribed_surface",
                "nearest_thermal_node_id": int(nnum[indices[index]]),
                "nearest_thermal_node_distance_m": float(distances[index]),
                "wall_temperature_K": float(wall_temperature[index]),
            }
            for index, point in enumerate(load.xy)
        ]
    incoming = load.incoming_flux(wall_temperature)
    radiation = (
        settings["emissivity"] * STEFAN_BOLTZMANN * (wall_temperature**4 - settings["trad"]**4)
        if settings["include_rad"] else np.zeros(len(load.xy))
    )
    if load.mode == "aerodynamic_surface_load":
        active_surface, driving_temperature = load.surface_active, load.gas_temperature
        return [
            {
                "time_s": float(time), "surface_order_index": index,
                "x_m": float(point[0]), "y_m": float(point[1]),
                "nearest_thermal_node_id": int(nnum[indices[index]]),
                "nearest_thermal_node_distance_m": float(distances[index]),
                "wall_temperature_K": float(wall_temperature[index]),
                "convective_heat_flux_into_solid_W_m2": float(incoming[index]),
                "radiative_heat_flux_out_W_m2": float(radiation[index]),
                "net_heat_flux_into_solid_W_m2": float(incoming[index] - radiation[index]),
                "aerodynamic_conductance_W_m2K": float(load.coefficient[index]),
                "effective_driving_temperature_K": float(driving_temperature[index]),
                "thermal_segment_active": int(load.segment_active[index]),
                "outside_calibration_temperature_range": bool(wall_temperature[index] < load.cold_temperature[index]
                                                               or wall_temperature[index] > load.equilibrium_temperature[index]),
            }
            for index, point in enumerate(load.xy) if active_surface[index]
        ]
    rows = [
        {
            "time_s": float(time), "surface_order_index": index,
            "x_m": float(point[0]), "y_m": float(point[1]),
            "applied_heat_flux_into_wall_W_m2": float(incoming[index]),
            "radiative_heat_flux_out_W_m2": float(radiation[index]),
            "net_surface_heat_flux_into_wall_W_m2": float(incoming[index] - radiation[index]),
            "nearest_thermal_node_id": int(nnum[indices[index]]),
            "nearest_thermal_node_distance_m": float(distances[index]),
            "wall_temperature_K": float(wall_temperature[index]),
        }
        for index, point in enumerate(load.xy)
    ]
    if load.mode == "convection":
        coefficient = np.broadcast_to(np.asarray(load.coefficient), (len(load.xy),))
        for index, row in enumerate(rows):
            row.update(recovery_temperature_K=float(load.gas_temperature[index]),
                       heat_transfer_coefficient_W_m2K=float(coefficient[index]))
    return rows


def write_csv(path, rows):
    if not rows:
        raise ThermalAnalysisError(f"Refusing to save empty result CSV: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(output, load, nnum, xyz, temperatures, times, settings, mesh_size, max_frames=60):
    """Save every wall result set and selected nodal frames for later reporting."""
    xyz, nnum = np.asarray(xyz), np.asarray(nnum)
    times = np.asarray(times, dtype=float)
    if len(temperatures) != len(times) or not len(times):
        raise ThermalAnalysisError("Thermal temperatures and result times do not match.")
    if not np.isfinite(times).all() or (times <= 0).any() or (np.diff(times) <= 0).any():
        raise ThermalAnalysisError("Saved MAPDL times must be finite, positive, and strictly increasing.")
    if not math.isclose(float(times[-1]), settings["duration"], rel_tol=1e-6, abs_tol=1e-8):
        raise ThermalAnalysisError("MAPDL did not reach the configured final time.")
    if xyz.shape != (len(nnum), 3):
        raise ThermalAnalysisError("Thermal node coordinates do not match node IDs.")
    indices, distances = map_wall_nodes(load.xy, xyz[:, :2], mesh_size)
    initial = np.full(len(nnum), settings["initial_t"])
    history = [{"time_s": 0.0, **temperature_stats(initial)}]
    wall_history = wall_rows(0, load, initial[indices], nnum, indices, distances, settings)
    snapshots = [{"result_set": 0, "time_s": 0.0, "temperature_K": initial}]
    selected = select_snapshots(len(times), max_frames)
    prescribed = load.mode == "prescribed_temperature"
    aerodynamic = load.mode == "aerodynamic_surface_load"
    surface_power = [] if prescribed or aerodynamic else [[0.0, integrate_boundary(load.xy, np.asarray([
        row["net_surface_heat_flux_into_wall_W_m2"] for row in wall_history]))]]
    final = None
    for number, (time, values) in enumerate(zip(times, temperatures), 1):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(nnum),):
            raise ThermalAnalysisError(f"Result set {number} contains the wrong number of nodal temperatures.")
        history.append({"time_s": float(time), **temperature_stats(values)})
        if prescribed and not np.allclose(values[indices], load.surface_temperature, rtol=1e-5, atol=1e-4):
            raise ThermalAnalysisError("MAPDL wall temperatures do not match the prescribed CFD surface field")
        rows = wall_rows(time, load, values[indices], nnum, indices, distances, settings)
        wall_history.extend(rows)
        if not prescribed and not aerodynamic:
            surface_power.append([float(time), integrate_boundary(load.xy, np.asarray([
                row["net_surface_heat_flux_into_wall_W_m2"] for row in rows]))])
        if number in selected:
            snapshots.append({"result_set": number, "time_s": float(time), "temperature_K": values.copy()})
        final = values
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    net_heat_load = None
    if aerodynamic:
        from .aerothermal_results import save_surface_energy
        net_heat_load = save_surface_energy(output, load, wall_history, [initial, *temperatures], times, settings, write_csv)
    write_csv(output / "temperature_history.csv", history)
    write_csv(output / "wall_temperature_history.csv", wall_history)
    write_csv(output / "thermal_nodes_final.csv", [
        {"node_id": int(nid), "x_m": float(point[0]), "y_m": float(point[1]), "z_m": float(point[2]), "temperature_K": float(value)}
        for nid, point, value in zip(nnum, xyz, final)
    ])
    final_wall = []
    wall_sample_count = int(np.count_nonzero(load.surface_active)) if aerodynamic else len(load.xy)
    for row in wall_history[-wall_sample_count:]:
        row = dict(row)
        row["thermal_wall_temperature_K"] = row["wall_temperature_K"]
        final_wall.append(row)
    write_csv(output / "wall_temperature_final.csv", final_wall)
    boundary_data = ({"boundary_aerodynamic_conductance_W_m2K": load.coefficient,
                      "boundary_aerodynamic_source_W_m2": load.aerodynamic_source,
                      "boundary_effective_driving_temperature_K": load.gas_temperature,
                      "boundary_thermal_segment_active": load.segment_active,
                      "boundary_thermal_active": load.surface_active} if aerodynamic else
                     {"boundary_prescribed_temperature_K": load.surface_temperature} if prescribed else
                     {"boundary_gas_temperature_K":load.gas_temperature, "convection_coefficient_W_m2K":load.coefficient}
                     if load.mode == "convection" else {"boundary_flux_W_m2":load.cold_flux})
    np.savez_compressed(output / "thermal_snapshots.npz", node_ids=nnum, coordinates_m=xyz,
        boundary_xy_m=load.xy, time_s=[snap["time_s"] for snap in snapshots],
        temperatures_K=np.asarray([snap["temperature_K"] for snap in snapshots]), **boundary_data)
    if not prescribed and not aerodynamic:
        power = np.asarray(surface_power)
        net_heat_load = float(np.sum(0.5 * (power[1:, 1] + power[:-1, 1]) * np.diff(power[:, 0])))
    return history, snapshots, temperature_stats(final), net_heat_load


def apply_surface_loads(mapdl, lines, load, settings):
    if load.mode == "prescribed_temperature":
        if settings["include_rad"]:
            raise ThermalAnalysisError("Prescribed temperature cannot also apply Mechanical radiation on the same boundary")
        return  # D,TEMP is assigned to all boundary nodes after meshing.
    convective = load.mode in {"convection", "aerodynamic_surface_load"}
    coefficients = np.broadcast_to(np.asarray(load.coefficient), (len(load.xy),)) if convective else None
    driving_temperature = load.gas_temperature if convective else None
    for index, line in enumerate(lines):
        if load.mode == "aerodynamic_surface_load" and not load.segment_active[index]:
            continue  # A retained solid cut with no surface load is naturally insulated.
        next_index = (index + 1) % len(lines)
        if convective:
            mapdl.sfl(line, "CONV", float(coefficients[index]), float(coefficients[next_index]),
                      float(driving_temperature[index]), float(driving_temperature[next_index]))
        else:  # Historical direct-flux utilities, never selected by the new runner.
            mapdl.sfl(line, "HFLUX", float(load.cold_flux[index]), float(load.cold_flux[next_index]))
        if settings["include_rad"]:
            mapdl.sfl(line, "RAD", settings["emissivity"], settings["emissivity"], settings["trad"], settings["trad"])


def apply_prescribed_temperature(mapdl, lines, load, nnum, xyz, mesh_size):
    """Constrain every meshed line node, not only CFD boundary keypoints."""
    from .prescribed_loading import interpolate_boundary_temperature
    mapdl.lsel("NONE")
    for line in lines:
        mapdl.lsel("A", "LINE", "", int(line))
    # NSLL NKEY=1 includes line-interior nodes and endpoint/keypoint nodes.
    # AMESH associates midside nodes with the solid-model boundary lines too.
    mapdl.nsll("S", 1)
    selected = np.asarray(mapdl.mesh.nnum, dtype=int).copy()
    mapdl.allsel()
    nnum, xyz = np.asarray(nnum, dtype=int), np.asarray(xyz, dtype=float)
    if not len(selected) or len(set(selected)) != len(selected):
        raise ThermalAnalysisError("MAPDL did not identify a unique set of solid-boundary nodes")
    index_by_node = {int(node): index for index, node in enumerate(nnum)}
    try:
        indices = np.asarray([index_by_node[int(node)] for node in selected], dtype=int)
    except KeyError as exc:
        raise ThermalAnalysisError("Selected boundary nodes do not match the thermal mesh") from exc
    tolerance = max(1e-9, float(mesh_size) * 1e-3)
    values = interpolate_boundary_temperature(load, xyz[indices, :2], tolerance)
    keypoint_indices, _ = map_wall_nodes(load.xy, xyz[:, :2], mesh_size)
    if not set(map(int, nnum[keypoint_indices])) <= set(map(int, selected)):
        raise ThermalAnalysisError("MAPDL boundary selection omitted a CFD wall keypoint")
    # Batch native D commands when supported to avoid a separate RPC for each
    # midside or refined line node, while retaining normal behavior for adapters.
    from contextlib import nullcontext
    with getattr(mapdl, "non_interactive", nullcontext()):
        for node, value in zip(selected, values):
            mapdl.d(int(node), "TEMP", float(value))
    load.constrained_node_count = len(selected)
    _progress(f"Prescribed CFD temperatures on {len(selected)} solid boundary nodes")
    return selected, values


def solve_thermal(mapdl, load, settings, mesh_size):
    _progress(f"Reconstructing solid from {len(load.xy)} CFD boundary vertices")
    mapdl.clear()
    mapdl.prep7()
    mapdl.et(1, "PLANE293")
    mapdl.keyopt(1, 1, 1)  # six-node triangle, unit out-of-plane thickness
    for label, value in (("KXX", settings["k"]), ("KYY", settings["k"]), ("DENS", settings["rho"]), ("C", settings["cp"])):
        mapdl.mp(label, 1, value)
    points = []
    for number, (x, y) in enumerate(load.xy, 1):
        points.append(int(mapdl.k("", float(x), float(y), 0)))
        if number % 50 == 0 or number == len(load.xy):
            _progress(f"Created {number}/{len(load.xy)} solid keypoints")
    lines = []
    for index, point in enumerate(points):
        lines.append(int(mapdl.l(point, points[(index + 1) % len(points)])))
        if (index + 1) % 50 == 0 or index + 1 == len(points):
            _progress(f"Created {index + 1}/{len(points)} boundary lines")
    mapdl.allsel()
    area = int(mapdl.al("ALL"))
    if area <= 0:
        raise ThermalAnalysisError("MAPDL could not reconstruct the solid polygon.")
    apply_surface_loads(mapdl, lines, load, settings)
    mapdl.toffst(0)  # all temperatures are Kelvin
    if settings["include_rad"]:
        mapdl.stef(STEFAN_BOLTZMANN)
    mapdl.esize(mesh_size)
    mapdl.mshape(1, "2D")
    mapdl.mshkey(0)
    _progress(f"Meshing solid; target element size={mesh_size:g} m")
    mapdl.amesh(area, verbose=True)
    mapdl.allsel()
    mapdl.sftran()
    nnum = np.asarray(mapdl.mesh.nnum, dtype=int).copy()
    xyz = np.asarray(mapdl.mesh.nodes, dtype=float).copy()
    n_elements = len(mapdl.mesh.enum)
    if len(nnum) < 3 or n_elements < 1:
        raise ThermalAnalysisError("MAPDL generated an empty thermal mesh.")
    _progress(f"Mesh complete: {len(nnum)} nodes / {n_elements} elements")
    if load.mode == "prescribed_temperature":
        _progress("Mapping the prescribed surface-temperature field onto the solid mesh")
        apply_prescribed_temperature(mapdl, lines, load, nnum, xyz, mesh_size)
    elif load.mode == "aerodynamic_surface_load":
        from .aerothermal_loading import prepare_boundary_quadrature
        count = prepare_boundary_quadrature(mapdl, lines, load, nnum, xyz, mesh_size)
        _progress(f"Implicit temperature-dependent gas heating plus radiation on {count} solved boundary nodes; no TEMP constraints")
    mapdl.finish()
    mapdl.slashsolu()
    mapdl.antype("TRANS")
    mapdl.trnopt("FULL")
    mapdl.timint("ON", "THERM")
    mapdl.tunif(settings["initial_t"])
    mapdl.kbc(1)
    mapdl.autots("ON")
    mapdl.time(settings["duration"])
    mapdl.deltim(settings["dt0"], settings["dtmin"], settings["dtmax"])
    mapdl.outres("ALL", "ALL")
    mapdl.neqit(100)
    _progress(f"Solving {len(nnum)} nodes / {n_elements} elements for {settings['duration']:g} s; streaming MAPDL output")
    message = str(mapdl.solve(verbose=True))
    if "*** ERROR ***" in message.upper():
        raise ThermalAnalysisError("MAPDL solve failed:\n" + message[-5000:])
    mapdl.finish()
    mapdl.post1()
    # Open the result file before querying ACTIVE,SET,NSET metadata.
    mapdl.set("LAST")
    nsets = int(mapdl.post_processing.nsets)
    times = list(mapdl.post_processing.time_values)
    if nsets < 1 or len(times) != nsets:
        raise ThermalAnalysisError("MAPDL result-set/time metadata is incomplete.")
    _progress(f"Reading {nsets} thermal result sets")
    temperatures = []
    for number in range(1, nsets + 1):
        mapdl.set(1, number)  # single transient load step
        temperatures.append(np.asarray(mapdl.post_processing.nodal_temperature(), dtype=float).copy())
        if number % 10 == 0 or number == nsets:
            _progress(f"Read result set {number}/{nsets}; physical time={float(times[number - 1]):g} s")
    return nnum, xyz, temperatures, times, n_elements


def run_thermal_analysis(run_dir=None, cfd_config="config.yaml", thermal_config="thermal_analysis.yaml"):
    from .config import load_config
    from .convection_loading import validate_settings, read_load
    from .gas_reporting import temperature_complete
    from .temperature_export import temperature_paths
    from .solution_quality import require_usable_solution
    from .utils import atomic_json, file_sha256
    _progress("Validating saved CFD temperatures and Mechanical configuration")
    cfg = load_config(cfd_config)
    ta_cfg = load_yaml(thermal_config)
    thermal_config_sha256 = file_sha256(thermal_config)
    settings = validate_settings(cfg, ta_cfg)  # Validate the loading law before any Ansys launch.
    launch_timeout = positive("mapdl_start_timeout_s", ta_cfg.get("mapdl_start_timeout_s", 120))
    work = Path(cfg["project"]["work_dir"])
    if run_dir is not None and Path(run_dir).resolve() != work:
        raise ThermalAnalysisError("--run-dir disagrees with the selected CFD config")
    endpoint = cfg.get("workflow", {}).get("strategy") == "two_endpoint"
    prescribed = settings["loading_mode"] == "prescribed_temperature"
    aerodynamic = settings["loading_mode"] == "aerodynamic_surface_load"
    if aerodynamic:
        from .aerothermal_calibration import verify_surface_package, calibration_paths
        from .aerothermal_loading import read_load as read_aerothermal_load
        boundary, metadata = verify_surface_package(cfg)
        load = read_aerothermal_load(boundary, scope=settings["thermal_surface_selection"])
        surface_package_sha256 = file_sha256(calibration_paths(cfg)["summary"])
        radiation = metadata["radiation"]
        if not (math.isclose(float(radiation["emissivity"]), settings["emissivity"], rel_tol=1e-10, abs_tol=1e-12)
                and math.isclose(float(radiation["external_radiation_temperature_K"]), settings["trad"], rel_tol=1e-10, abs_tol=1e-12)):
            raise ThermalAnalysisError("Mechanical radiation settings disagree with the verified CFD calibration package.")
    elif endpoint:
        from .endpoint_loading import verify_convection_package, convection_paths
        boundary, metadata = verify_convection_package(cfg)
        actual_minimum = metadata["diagnostics"]["minimum_absolute_delta_temperature_K"]
        if actual_minimum < settings["min_delta_temperature_k"]:
            raise ThermalAnalysisError("Endpoint temperature separation is below loading.min_delta_temperature_k; "
                                       "the coefficient is not identifiable at the requested tolerance")
        load = read_load(boundary)
        package_sha256 = file_sha256(convection_paths(cfg)[1])
    else:
        quality = require_usable_solution(cfg)
        if quality["status"] != "residual_screen_passed":
            raise ThermalAnalysisError("Mechanical requires a CFD residual screen with no final temperature/pressure limiting.")
        if not temperature_complete(cfg):
            raise ThermalAnalysisError("Temperature export is missing or stale; rerun study.py export first.")
        metadata = json.loads(temperature_paths(cfg)[1].read_text())
        temperature_summary_sha256 = file_sha256(temperature_paths(cfg)[1])
        node_metadata = metadata.get("wall_nodes", {})
        boundary = temperature_paths(cfg)[0].parent / "wall_temperature_nodes.csv"
        if not boundary.is_file():raise ThermalAnalysisError("Missing wall-node temperatures; rerun CFD export.")
        if node_metadata.get("csv_sha256") != file_sha256(boundary):
            raise ThermalAnalysisError("CFD boundary-node temperature CSV is missing or modified; rerun export.")
        if prescribed:
            from .prescribed_loading import read_load as read_prescribed_load
            load = read_prescribed_load(boundary)
        else:
            load = read_load(boundary, settings["h"])
    boundary_sha256 = file_sha256(boundary)
    mesh_size = auto_mesh_size(load.xy, ta_cfg.get("mesh_size_m"))
    output = work / "thermal_analysis"
    output.mkdir(parents=True, exist_ok=True)
    _progress(f"Case: {work}; loading={settings['loading_mode']}; {len(load.xy)} boundary vertices")
    _progress(f"Tungsten properties: k={settings['k']:g} W/(m K), rho={settings['rho']:g} kg/m3, Cp={settings['cp']:g} J/(kg K)")
    _progress(f"Exposure={settings['duration']:g} s; target mesh size={mesh_size:g} m")
    _progress(f"MAPDL command/output logs: {output}")
    from ansys.mapdl.core import launch_mapdl
    _progress(f"Launching Mechanical APDL {ta_cfg.get('mapdl_version', 251)}; connection timeout={launch_timeout:g} s")
    try:
        mapdl = launch_mapdl(version=int(ta_cfg.get("mapdl_version",251)), mode="grpc",
                             nproc=int(ta_cfg.get("processors",4)), run_location=str(output),
                             jobname="nose_thermal", start_timeout=launch_timeout,
                             log_apdl=str(output / "mapdl_commands.log"),
                             mapdl_output=str(output / "mapdl_console.out"))
    except Exception as exc:
        raise ThermalAnalysisError(f"MAPDL launch failed: {exc}. Native startup logs are in {output}") from exc
    _progress("Mechanical APDL connected")
    try:
        nnum, xyz, temperatures, times, n_elements = solve_thermal(mapdl,load,settings,mesh_size)
        _progress("Saving thermal histories and final node temperatures")
        history, snapshots, final, net_heat = save_results(output,load,nnum,xyz,temperatures,times,
            settings,mesh_size,ta_cfg.get("visualization",{}).get("max_animation_frames",60))
        viz = dict(ta_cfg.get("visualization",{}), save_heat_flux_image=False)
        if prescribed or aerodynamic:
            viz["loading_mode"] = load.mode
        from .reporting import create_thermal_visualizations
        _progress("Generating thermal plots")
        figures = create_thermal_visualizations(output,load.xy,np.zeros(len(load.xy)),xyz,history,snapshots,viz)
        loading_limits = (["Gas-side heating is an affine wall-temperature law calibrated from cold-wall and radiative-equilibrium CFD states; no surface temperature is prescribed.",
                           "MAPDL evaluates native CONV and RAD loads at the current solved boundary temperature within each nonlinear thermal iteration.",
                           "H and effective driving temperature are each linearly interpolated on each reconstructed straight boundary line; their product need not equal a linearly interpolated affine source between vertices.",
                           "The affine law is extrapolated without clipping outside its two-state calibration temperature range; the effective driving temperature is not a directly sampled CFD gas temperature.",
                           "Boundary energies use quadratic FE boundary-temperature interpolation and five-point spatial Gaussian quadrature, then trapezoidal integration over physical saved substeps.",
                           "Integrated surface energy is an independent postprocessed heat input, not a native constraint reaction or verified global internal-energy balance."] if aerodynamic else
                          ["The steady CFD radiative-equilibrium wall temperature is prescribed on every Mechanical boundary node.",
                           "The surface temperature steps to the imported field at t=0+ and remains fixed; only the interior temperature response is predicted.",
                           "No convection coefficient or boundary heat flux is inferred; prescribed-temperature reaction heat is not exported.",
                           "Radiation is included in the CFD equilibrium surface model and is not applied again on the prescribed Mechanical boundary."] if prescribed else
                          ["Steady matched CFD endpoints define spatial h and recovery temperature; both remain fixed throughout exposure.",
                           "The derived h is calibrated at the cold-wall temperature; wall-temperature dependence requires further calibration."] if endpoint else
                          ["Constant steady CFD recovery temperature and specified uniform h throughout exposure."])
        result_names = ["wall_temperature_history.csv", "thermal_nodes_final.csv", "temperature_history.csv"]
        if aerodynamic:
            result_names += ["surface_heat_flux_history.csv", "surface_energy_history.csv", "global_energy_history.csv",
                             "thermal_snapshots.npz", "wall_temperature_final.csv"]
        summary = dict(status="complete", model_version=AEROTHERMAL_MODEL_VERSION if aerodynamic else PRESCRIBED_MODEL_VERSION if prescribed else ENDPOINT_MODEL_VERSION if endpoint else MODEL_VERSION,
            loading_mode="aerodynamic_surface_load" if aerodynamic else "prescribed_temperature" if prescribed else "convective_from_cfd" if endpoint else "convection",
            backend="Mechanical APDL", material=ta_cfg["material"], analysis={**ta_cfg["transient"], "initial_temperature_k":settings["initial_t"]},
            thermal_config_path=str(Path(thermal_config).resolve()), thermal_config_sha256=thermal_config_sha256,
            result_sha256={name:file_sha256(output/name) for name in result_names},
            settings=settings, thermal_configuration=ta_cfg, boundary_csv_sha256=boundary_sha256,
            source_sha256=metadata["source_sha256"], final_temperature=final,
            thermal_mesh=dict(nodes=len(nnum),elements=n_elements,size_m=mesh_size),
            net_heat_load_per_unit_span_J_m=net_heat, visualization_settings=viz, figures=figures,
            limitations=[*loading_limits,
                         "Constant tungsten properties; no ablation, structural response or temperature dependence.",
                         "Reconstructed planar solid, unit out-of-plane thickness; not a full 3-D vehicle.",
                         "CFD is one-way coupled; the transient solid response does not feed back into the gas solution."])
        if prescribed:
            summary.update(source_temperature_summary_sha256=temperature_summary_sha256,
                           temperature_conditions=metadata["conditions"],
                           boundary_temperature_application="step_at_t0_plus",
                           temperature_history_sampling="all_solid_nodes_including_prescribed_surface",
                           initial_state="uniform initial temperature at t=0 before boundary application",
                           heat_transfer_status="constraint_reaction_not_exported")
            summary["thermal_mesh"]["prescribed_boundary_nodes"] = load.constrained_node_count
        if aerodynamic:
            excluded_segments = int(np.count_nonzero(~load.segment_active))
            if excluded_segments:
                summary["limitations"].append("The retained rear cut is insulated: no convection or radiation is applied there, and attachment conduction into an omitted vehicle is not modeled.")
            summary.update(surface_package_sha256=surface_package_sha256,
                           source_surface_load_summary_sha256=surface_package_sha256,
                           surface_load_conditions=metadata["conditions"],
                           surface_load_derivation=metadata["derivation"],
                           surface_load_diagnostics=metadata["diagnostics"],
                           source_surface_scope=metadata.get("thermal_scope", {}),
                           boundary_temperature_application="solved_with_implicit_CONV_and_RAD",
                           temperature_history_sampling="solved_wall_vertices_and_all_solid_nodes",
                           initial_state="uniform initial temperature; finite surface heating begins at t=0",
                           heat_transfer_status="postprocessed_temperature_dependent_convective_and_radiative_fluxes",
                           thermal_surface_scope=dict(active_segments=int(np.count_nonzero(load.segment_active)),
                                                      excluded_segments=excluded_segments,
                                                      active_wall_vertices=int(np.count_nonzero(load.surface_active)),
                                                      rear_boundary="insulated" if excluded_segments else "aerodynamically_loaded",
                                                      full_solid_temperature_field_retained=True),
                           energy_method={"local": "trapezoid_in_time_at_original_CFD_vertices",
                                          "global_space": "5_point_Gaussian_quadrature_on_all_quadratic_FE_boundary_edges",
                                          "global_time": "trapezoid_over_saved_physical_substeps_including_initial_flux",
                                          "units_local": "J/m2", "units_global": "J/m_per_unit_span",
                                          "native_internal_energy_balance_verified": False})
        if endpoint:
            summary.update(convection_package_sha256=package_sha256,
                           convection_conditions=metadata["conditions"],
                           convection_derivation=metadata["derivation"],
                           convection_diagnostics=metadata["diagnostics"],
                           convection_package_model_version=metadata["model_version"])
        atomic_json(output/"thermal_summary.json",summary)
        if prescribed:
            description = f"prescribed equilibrium surface T={float(np.min(load.surface_temperature)):g}–{float(np.max(load.surface_temperature)):g} K"
        elif aerodynamic:
            active_h = np.asarray(load.coefficient)[load.surface_active]
            description = f"calibrated H={float(np.min(active_h)):g}–{float(np.max(active_h)):g} W/(m² K), implicit radiation; surface temperature solved"
        else:
            coefficient_description = (f"spatial h={float(np.min(load.coefficient)):g}–{float(np.max(load.coefficient)):g}" if endpoint
                                       else f"h={settings['h']:g}")
            description = coefficient_description + " W/(m² K)"
        print(f"[thermal] Complete: {output}; duration={settings['duration']:g} s; {description}")
        return summary
    finally:
        _progress("Closing Mechanical APDL")
        mapdl.exit()
