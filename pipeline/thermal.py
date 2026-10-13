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
    prepare_load, read_boundary, validate_settings,
)
from .trade_study import thermal_screening_metrics

MODEL_VERSION = "3.0-temperature-convection"
STEFAN_BOLTZMANN = 5.670374419e-8


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
    incoming = load.incoming_flux(wall_temperature)
    radiation = (
        settings["emissivity"] * STEFAN_BOLTZMANN * (wall_temperature**4 - settings["trad"]**4)
        if settings["include_rad"] else np.zeros(len(load.xy))
    )
    return [
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
    surface_power = [[0.0, integrate_boundary(load.xy, np.asarray([row["net_surface_heat_flux_into_wall_W_m2"] for row in wall_history]))]]
    final = None
    for number, (time, values) in enumerate(zip(times, temperatures), 1):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(nnum),):
            raise ThermalAnalysisError(f"Result set {number} contains the wrong number of nodal temperatures.")
        history.append({"time_s": float(time), **temperature_stats(values)})
        rows = wall_rows(time, load, values[indices], nnum, indices, distances, settings)
        wall_history.extend(rows)
        surface_power.append([float(time), integrate_boundary(load.xy, np.asarray([row["net_surface_heat_flux_into_wall_W_m2"] for row in rows]))])
        if number in selected:
            snapshots.append({"result_set": number, "time_s": float(time), "temperature_K": values.copy()})
        final = values
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "temperature_history.csv", history)
    write_csv(output / "wall_temperature_history.csv", wall_history)
    write_csv(output / "thermal_nodes_final.csv", [
        {"node_id": int(nid), "x_m": float(point[0]), "y_m": float(point[1]), "z_m": float(point[2]), "temperature_K": float(value)}
        for nid, point, value in zip(nnum, xyz, final)
    ])
    final_wall = []
    for row in wall_history[-len(load.xy):]:
        row = dict(row)
        row["thermal_wall_temperature_K"] = row["wall_temperature_K"]
        final_wall.append(row)
    write_csv(output / "wall_temperature_final.csv", final_wall)
    boundary_data = ({"boundary_gas_temperature_K":load.gas_temperature, "convection_coefficient_W_m2K":load.coefficient}
                     if load.mode == "convection" else {"boundary_flux_W_m2":load.cold_flux})
    np.savez_compressed(output / "thermal_snapshots.npz", node_ids=nnum, coordinates_m=xyz,
        boundary_xy_m=load.xy, time_s=[snap["time_s"] for snap in snapshots],
        temperatures_K=np.asarray([snap["temperature_K"] for snap in snapshots]), **boundary_data)
    power = np.asarray(surface_power)
    net_heat_load = float(np.sum(0.5 * (power[1:, 1] + power[:-1, 1]) * np.diff(power[:, 0])))
    return history, snapshots, temperature_stats(final), net_heat_load


def apply_surface_loads(mapdl, lines, load, settings):
    for index, line in enumerate(lines):
        next_index = (index + 1) % len(lines)
        if load.mode == "convection":
            mapdl.sfl(line, "CONV", load.coefficient, load.coefficient,
                      float(load.gas_temperature[index]), float(load.gas_temperature[next_index]))
        else:  # Historical direct-flux utilities, never selected by the new runner.
            mapdl.sfl(line, "HFLUX", float(load.cold_flux[index]), float(load.cold_flux[next_index]))
        if settings["include_rad"]:
            mapdl.sfl(line, "RAD", settings["emissivity"], settings["emissivity"], settings["trad"], settings["trad"])


def solve_thermal(mapdl, load, settings, mesh_size):
    mapdl.clear()
    mapdl.prep7()
    mapdl.et(1, "PLANE293")
    mapdl.keyopt(1, 1, 1)  # six-node triangle, unit out-of-plane thickness
    for label, value in (("KXX", settings["k"]), ("KYY", settings["k"]), ("DENS", settings["rho"]), ("C", settings["cp"])):
        mapdl.mp(label, 1, value)
    points = [int(mapdl.k("", float(x), float(y), 0)) for x, y in load.xy]
    lines = [int(mapdl.l(point, points[(index + 1) % len(points)])) for index, point in enumerate(points)]
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
    mapdl.amesh(area)
    mapdl.allsel()
    mapdl.sftran()
    nnum = np.asarray(mapdl.mesh.nnum, dtype=int).copy()
    xyz = np.asarray(mapdl.mesh.nodes, dtype=float).copy()
    n_elements = len(mapdl.mesh.enum)
    if len(nnum) < 3 or n_elements < 1:
        raise ThermalAnalysisError("MAPDL generated an empty thermal mesh.")
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
    print(f"[thermal] Solving {len(nnum)} nodes / {n_elements} elements for {settings['duration']:g} s...")
    message = str(mapdl.solve())
    if "*** ERROR ***" in message.upper():
        raise ThermalAnalysisError("MAPDL solve failed:\n" + message[-5000:])
    mapdl.finish()
    mapdl.post1()
    nsets = int(mapdl.post_processing.nsets)
    times = list(mapdl.post_processing.time_values)
    if nsets < 1 or len(times) != nsets:
        raise ThermalAnalysisError("MAPDL result-set/time metadata is incomplete.")
    temperatures = []
    for number in range(1, nsets + 1):
        mapdl.set(1, number)  # single transient load step
        temperatures.append(np.asarray(mapdl.post_processing.nodal_temperature(), dtype=float).copy())
    return nnum, xyz, temperatures, times, n_elements


def run_thermal_analysis(run_dir=None, cfd_config="config.yaml", thermal_config="thermal_analysis.yaml"):
    from .config import load_config
    from .convection_loading import validate_settings, read_load
    from .gas_reporting import temperature_complete
    from .temperature_export import temperature_paths
    from .solution_quality import require_usable_solution
    from .utils import atomic_json, file_sha256
    cfg = load_config(cfd_config)
    ta_cfg = load_yaml(thermal_config)
    settings = validate_settings(cfg, ta_cfg)  # h is checked before any Ansys launch.
    work = Path(cfg["project"]["work_dir"])
    if run_dir is not None and Path(run_dir).resolve() != work:
        raise ThermalAnalysisError("--run-dir disagrees with the selected CFD config")
    quality = require_usable_solution(cfg)
    if quality["status"] != "residual_screen_passed":
        raise ThermalAnalysisError("Mechanical requires a CFD residual screen with no final temperature/pressure limiting.")
    if not temperature_complete(cfg):
        raise ThermalAnalysisError("Temperature export is missing or stale; rerun study.py export first.")
    metadata = json.loads(temperature_paths(cfg)[1].read_text())
    node_metadata = metadata.get("wall_nodes", {})
    boundary = temperature_paths(cfg)[0].parent / "wall_temperature_nodes.csv"
    if not boundary.is_file():raise ThermalAnalysisError("Missing wall-node temperatures; rerun CFD export.")
    if node_metadata.get("csv_sha256") != file_sha256(boundary):
        raise ThermalAnalysisError("CFD boundary-node temperature CSV is missing or modified; rerun export.")
    load = read_load(boundary, settings["h"])
    mesh_size = auto_mesh_size(load.xy, ta_cfg.get("mesh_size_m"))
    output = work / "thermal_analysis"
    output.mkdir(parents=True, exist_ok=True)
    from ansys.mapdl.core import launch_mapdl
    mapdl = launch_mapdl(version=int(ta_cfg.get("mapdl_version",251)),
                         nproc=int(ta_cfg.get("processors",4)), run_location=str(output),
                         jobname="nose_thermal")
    try:
        nnum, xyz, temperatures, times, n_elements = solve_thermal(mapdl,load,settings,mesh_size)
        history, snapshots, final, net_heat = save_results(output,load,nnum,xyz,temperatures,times,
            settings,mesh_size,ta_cfg.get("visualization",{}).get("max_animation_frames",60))
        viz = dict(ta_cfg.get("visualization",{}), save_heat_flux_image=False)
        from .reporting import create_thermal_visualizations
        figures = create_thermal_visualizations(output,load.xy,np.zeros(len(load.xy)),xyz,history,snapshots,viz)
        summary = dict(status="complete", model_version=MODEL_VERSION, loading_mode="convection",
            backend="Mechanical APDL", material=ta_cfg["material"], analysis={**ta_cfg["transient"], "initial_temperature_k":settings["initial_t"]},
            thermal_config_path=str(Path(thermal_config).resolve()), thermal_config_sha256=file_sha256(thermal_config),
            result_sha256={name:file_sha256(output/name) for name in ("wall_temperature_history.csv", "thermal_nodes_final.csv", "temperature_history.csv")},
            settings=settings, thermal_configuration=ta_cfg, boundary_csv_sha256=file_sha256(boundary),
            source_sha256=metadata["source_sha256"], final_temperature=final,
            thermal_mesh=dict(nodes=len(nnum),elements=n_elements,size_m=mesh_size),
            net_heat_load_per_unit_span_J_m=net_heat, visualization_settings=viz, figures=figures,
            limitations=["Constant steady CFD recovery temperature and specified uniform h throughout exposure.",
                         "Constant tungsten properties; no ablation, structural response or temperature dependence.",
                         "Reconstructed planar solid, unit out-of-plane thickness; not a full 3-D vehicle.",
                         "Adiabatic CFD is one-way coupled; heating does not feed back into the gas solution."])
        atomic_json(output/"thermal_summary.json",summary)
        print(f"[thermal] Complete: {output}; duration={settings['duration']:g} s; h={settings['h']:g} W/(m² K)")
        return summary
    finally:
        mapdl.exit()
