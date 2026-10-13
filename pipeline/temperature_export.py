"""Adiabatic wall and fluid-side static temperatures from a solved Fluent case."""
from pathlib import Path
import csv
import os
import tempfile

import numpy as np

from .thermal_export import order_segments, unpack_faces
from .utils import atomic_json


def temperature_paths(cfg):
    directory = Path(cfg.get("temperature_export", {}).get("output_directory", "temperature"))
    if not directory.is_absolute():
        directory = Path(cfg["project"]["work_dir"]) / directory
    return directory / "wall_temperatures.csv", directory / "temperature_summary.json"


def temperature_conditions(cfg):
    flight = cfg["flight"]
    direction = np.asarray(flight["flow_direction"], dtype=float)
    magnitude = float(np.linalg.norm(direction))
    if direction.shape != (2,) or not np.isfinite(direction).all() or magnitude == 0:
        raise ValueError("Invalid freestream direction")
    return dict(flight=dict(mach=float(flight["mach"]), static_pressure_pa=float(flight["static_pressure_pa"]),
        static_temperature_k=float(flight["static_temperature_k"]), flow_direction=(direction / magnitude).tolist()),
        physics=cfg.get("physics", {}), thermal=cfg["thermal"], mesh=cfg.get("mesh", {}),
        geometry={key:value for key,value in cfg["geometry"].items() if key != "input_file"})


def atomic_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()


def export_temperatures(solver, cfg, wall_zones):
    from ansys.fluent.core.fields.field_data_interfaces import (
        ScalarFieldDataRequest, SurfaceFieldDataRequest, SurfaceDataType,
    )
    if cfg["thermal"] != {"mode": "adiabatic"}:
        raise ValueError("Temperature export requires the adiabatic no-heat-transfer model")
    base = cfg["geometry"].get("body_wall_name", "body_wall")
    if not wall_zones or any(z != base and not z.startswith(base + ".") for z in wall_zones):
        raise ValueError("Select only the named body-wall zones")
    settings = cfg.get("temperature_export", {})
    field = settings.get("static_temperature_field", "temperature")
    data = solver.fields.field_data
    available = data.scalar_fields.allowed_values()
    for required in (field,):
        if required not in available:
            raise ValueError(f"Field {required!r} unavailable. Run study.py export --inspect-fields.")
    geometry = data.get_field_data(SurfaceFieldDataRequest(surfaces=wall_zones,
        data_types=[SurfaceDataType.Vertices, SurfaceDataType.FacesConnectivity, SurfaceDataType.FacesCentroid],
        flatten_connectivity=True))
    def scalar(boundary, name):
        return data.get_field_data(ScalarFieldDataRequest(
            surfaces=wall_zones, field_name=name, node_value=False, boundary_value=boundary))
    wall, fluid = scalar(True, field), scalar(False, field)
    nodal = data.get_field_data(ScalarFieldDataRequest(
        surfaces=wall_zones, field_name=field, node_value=True, boundary_value=True))
    records = []
    for zone in wall_zones:
        surface = geometry[zone]
        vertices = np.asarray(surface.vertices, dtype=float)
        centers = np.asarray(surface.face_centroids, dtype=float)
        faces = unpack_faces(surface.connectivity)
        arrays = [np.asarray(value[zone], dtype=float).reshape(-1) for value in (wall, fluid)]
        node_t = np.asarray(nodal[zone], dtype=float).reshape(-1)
        if len(node_t) != len(vertices) or not np.isfinite(node_t).all() or np.any(node_t <= 0):
            raise ValueError(f"Invalid wall-node temperature values on {zone}")
        if any(len(value) != len(faces) for value in [centers, *arrays]):
            raise ValueError(f"Temperature/geometry face counts differ on {zone}")
        if not all(np.isfinite(value).all() for value in [vertices, centers, *arrays]):
            raise ValueError("Nonfinite temperature or geometry")
        if any(np.any(value <= 0) for value in arrays[:2]):
            raise ValueError("Static temperatures must be positive kelvin")
        for index, face in enumerate(faces):
            if len(face) != 2 or np.any(face < 0) or np.any(face >= len(vertices)):
                raise ValueError("Expected valid planar line faces")
            start, end = vertices[face[0]], vertices[face[1]]
            if max(abs(start[2]), abs(end[2])) > 1e-7:
                raise ValueError("Body wall must lie in the XY plane")
            records.append(dict(start=start, end=end, zone_name=zone, face_index=index,
                center=centers[index], wall_temperature_K=float(arrays[0][index]),
                near_wall_fluid_temperature_K=float(arrays[1][index]),
                start_temperature_K=float(node_t[face[0]]), end_temperature_K=float(node_t[face[1]])))
    records = order_segments(records, float(settings.get("topology_tolerance_m", 1e-9)))
    rows, distance = [], 0.0
    for index, record in enumerate(records):
        length = float(np.linalg.norm(record["end"] - record["start"]))
        x, y, z = record["center"]
        rows.append(dict(surface_order_index=index, zone_name=record["zone_name"], face_index=record["face_index"],
            x_m=float(x), y_m=float(y), z_m=float(z), s_m=distance + length / 2,
            segment_length_m=length, wall_temperature_K=record["wall_temperature_K"],
            near_wall_fluid_temperature_K=record["near_wall_fluid_temperature_K"]))
        distance += length
    csv_path, summary_path = temperature_paths(cfg)
    def stats(column):
        values = np.asarray([row[column] for row in rows])
        weights = np.asarray([row["segment_length_m"] for row in rows])
        return dict(minimum_K=float(values.min()), maximum_K=float(values.max()),
                    length_weighted_mean_K=float(np.average(values, weights=weights)))
    summary = dict(status="complete", thermal_mode="adiabatic", wall_temperatures_csv=str(csv_path),
        temperature_export_signature=settings,
        conditions=temperature_conditions(cfg),
        face_count=len(rows), body_wall_zones=list(wall_zones), static_temperature_field=field,
        wall_temperature=stats("wall_temperature_K"), near_wall_fluid_temperature=stats("near_wall_fluid_temperature_K"),
        sampling=dict(wall="node_value=False, boundary_value=True",
                      fluid_side="node_value=False, boundary_value=False",
                      coordinates="Wall-face centroids; not fluid-cell centers or a specified offset distance"),
        limitations=["Fluid-side surface values depend on Fluent's boundary-value support and the near-wall mesh.",
                     "Wall values are adiabatic recovery temperatures, not transient solid temperatures."])
    atomic_csv(csv_path, rows)
    # Follow connectivity and keep actual CFD vertices, not face centroids, for
    # reconstructing the solid and mapping the convection recovery temperature.
    current = min((p for r in records for p in (r["start"], r["end"])), key=lambda p: tuple(p[:2]))
    node_rows = []
    for index, record in enumerate(records):
        forward = np.linalg.norm(current - record["start"]) <= float(settings.get("topology_tolerance_m", 1e-9))
        point = record["start"] if forward else record["end"]
        temperature = record["start_temperature_K"] if forward else record["end_temperature_K"]
        node_rows.append(dict(surface_order_index=index, x_m=float(point[0]), y_m=float(point[1]),
                              z_m=float(point[2]), recovery_temperature_K=temperature))
        current = record["end"] if forward else record["start"]
    node_path = csv_path.parent / "wall_temperature_nodes.csv"
    atomic_csv(node_path, node_rows)
    from .utils import file_sha256
    summary["wall_nodes"] = dict(status="complete", csv=str(node_path), point_count=len(node_rows),
                                  csv_sha256=file_sha256(node_path), sampling="Fluent boundary nodal temperature")
    if settings.get("export_fluid_nodes", True):
        try:
            interiors = solver.settings.setup.boundary_conditions.interior.get_object_names()
            if not interiors:raise ValueError("No interior surfaces available for fluid-node export")
            mesh = data.get_field_data(SurfaceFieldDataRequest(
                surfaces=interiors, data_types=[SurfaceDataType.Vertices]))
            temperatures = data.get_field_data(ScalarFieldDataRequest(
                surfaces=interiors, field_name=field, node_value=True, boundary_value=False))
            nodes = []
            for zone in interiors:
                xyz = np.asarray(mesh[zone].vertices, dtype=float)
                values = np.asarray(temperatures[zone], dtype=float).reshape(-1)
                if len(xyz) != len(values) or not np.isfinite(xyz).all() or not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError(f"Invalid fluid-node values on {zone}")
                nodes.extend(dict(zone_name=zone, vertex_index=i, x_m=float(p[0]), y_m=float(p[1]),
                    z_m=float(p[2]), static_temperature_K=float(t)) for i, (p, t) in enumerate(zip(xyz, values)))
            if not nodes:raise ValueError("No fluid-node temperatures returned")
            node_path = csv_path.parent / "fluid_temperature_nodes.csv"
            atomic_csv(node_path, nodes)
            summary["fluid_nodes"] = dict(status="complete", csv=str(node_path), point_count=len(nodes))
        except (ValueError, RuntimeError, AttributeError, KeyError) as exc:
            (csv_path.parent / "fluid_temperature_nodes.csv").unlink(missing_ok=True)
            summary["fluid_nodes"] = dict(status="unavailable", error=str(exc))
            print(f"[temperature] Optional fluid-node export unavailable: {exc}", flush=True)
    atomic_json(summary_path, summary)
    return summary
