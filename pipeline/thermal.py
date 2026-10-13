"""Historical solid-result utilities; solid thermal execution is disabled."""
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

MODEL_VERSION = "2.0-direct-heat-flux"
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
    np.savez_compressed(output / "thermal_snapshots.npz", node_ids=nnum, coordinates_m=xyz, boundary_xy_m=load.xy, boundary_flux_W_m2=load.cold_flux, time_s=[snap["time_s"] for snap in snapshots], temperatures_K=np.asarray([snap["temperature_K"] for snap in snapshots]))
    power = np.asarray(surface_power)
    net_heat_load = float(np.sum(0.5 * (power[1:, 1] + power[:-1, 1]) * np.diff(power[:, 0])))
    return history, snapshots, temperature_stats(final), net_heat_load


def apply_surface_loads(mapdl, lines, load, settings):
    for index, line in enumerate(lines):
        next_index = (index + 1) % len(lines)
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
    raise ThermalAnalysisError("Solid thermal analysis is disabled for this adiabatic study. "
                               "Use study.py export for gas and wall temperatures.")
