"""Face-collocated endpoint exports, with actual vertices for solid geometry."""
from pathlib import Path
import numpy as np

from .temperature_export import atomic_csv, temperature_conditions
from .thermal_export import order_segments, unpack_faces
from .utils import atomic_json, file_sha256, work_paths


def export_endpoint(solver, cfg, zones):
    from ansys.fluent.core.fields.field_data_interfaces import (
        ScalarFieldDataRequest, SurfaceFieldDataRequest, SurfaceDataType,
    )
    cold = cfg["thermal"]["mode"] == "fixed_temperature"
    settings = cfg.get("thermal_export", {})
    temperature = settings.get("temperature_field_name", "temperature")
    flux = settings.get("heat_flux_field_name", "heat-flux")
    sign = settings.get("heat_flux_into_wall_sign")
    if cold and (isinstance(sign, bool) or sign not in (-1, 1)):
        raise ValueError("thermal_export.heat_flux_into_wall_sign must explicitly be -1 or +1")
    data = solver.fields.field_data
    available = data.scalar_fields.allowed_values()
    for field in (temperature, flux) if cold else (temperature,):
        if field not in available:raise ValueError(f"Endpoint field {field!r} unavailable; inspect fields before exporting")
    geometry = data.get_field_data(SurfaceFieldDataRequest(surfaces=zones,
        data_types=[SurfaceDataType.Vertices, SurfaceDataType.FacesConnectivity, SurfaceDataType.FacesCentroid],
        flatten_connectivity=True))
    def values(field, nodes=False):
        return data.get_field_data(ScalarFieldDataRequest(surfaces=zones, field_name=field,
            node_value=nodes, boundary_value=True))
    face_t, node_t = values(temperature), values(temperature, True)
    face_q = values(flux) if cold else None
    records = []
    for zone in zones:
        surface = geometry[zone]
        vertices = np.asarray(surface.vertices, dtype=float)
        centers = np.asarray(surface.face_centroids, dtype=float)
        faces = unpack_faces(surface.connectivity)
        temperatures = np.asarray(face_t[zone], dtype=float).reshape(-1)
        nodal = np.asarray(node_t[zone], dtype=float).reshape(-1)
        heat_flux = np.asarray(face_q[zone], dtype=float).reshape(-1) if cold else None
        arrays = [vertices, centers, temperatures, nodal] + ([heat_flux] if cold else [])
        if not all(np.isfinite(array).all() for array in arrays):raise ValueError("Nonfinite endpoint field/geometry")
        if (len(nodal) != len(vertices) or len(temperatures) != len(faces) or len(centers) != len(faces)
                or (cold and len(heat_flux) != len(faces))):
            raise ValueError(f"Endpoint geometry and field counts differ on {zone}")
        if np.any(temperatures <= 0) or np.any(nodal <= 0):raise ValueError("Endpoint temperatures must be positive kelvin")
        for index, face in enumerate(faces):
            if len(face) != 2 or np.any(face < 0) or np.any(face >= len(vertices)):
                raise ValueError("Expected planar line faces at the body wall")
            start, end = vertices[face[0]], vertices[face[1]]
            if max(abs(start[2]), abs(end[2])) > 1e-7:raise ValueError("Body wall must lie in the XY plane")
            records.append(dict(start=start, end=end, center=centers[index], zone_name=zone,
                face_index=index, wall_temperature_K=float(temperatures[index]),
                start_temperature_K=float(nodal[face[0]]), end_temperature_K=float(nodal[face[1]]),
                **({"raw_heat_flux_W_m2": float(heat_flux[index])} if cold else {})))
    tolerance = float(cfg.get("workflow", {}).get("coordinate_tolerance_m", 1e-9))
    records = order_segments(records, tolerance)
    current = min((p for r in records for p in (r["start"], r["end"])), key=lambda p:tuple(p[:2]))
    rows, nodes = [], []
    for index, record in enumerate(records):
        forward = np.linalg.norm(current - record["start"]) <= tolerance
        start, end = (record["start"], record["end"]) if forward else (record["end"], record["start"])
        length = float(np.linalg.norm(end-start))
        center = record["center"]
        row = dict(surface_order_index=index, zone_name=record["zone_name"], face_index=record["face_index"],
            x_m=float(center[0]), y_m=float(center[1]), z_m=float(center[2]), segment_length_m=length,
            start_x_m=float(start[0]), start_y_m=float(start[1]), end_x_m=float(end[0]), end_y_m=float(end[1]),
            wall_temperature_K=record["wall_temperature_K"])
        if cold:row.update(raw_heat_flux_W_m2=record["raw_heat_flux_W_m2"],
                           heat_flux_into_wall_W_m2=sign*record["raw_heat_flux_W_m2"])
        rows.append(row)
        nodes.append(dict(surface_order_index=index, x_m=float(start[0]), y_m=float(start[1]), z_m=float(start[2]),
            wall_temperature_K=record["start_temperature_K"] if forward else record["end_temperature_K"]))
        current = end
    paths = work_paths(cfg)
    directory = paths["work"] / "thermal_bc"
    faces_path, nodes_path = directory/"wall_faces.csv", directory/"wall_nodes.csv"
    atomic_csv(faces_path, rows); atomic_csv(nodes_path, nodes)
    root = Path(cfg.get("_study_work_dir", cfg["project"]["work_dir"]))
    summary = dict(status="complete", thermal_mode=cfg["thermal"]["mode"], conditions=temperature_conditions(cfg),
        mesh_sha256=file_sha256(root/"02_mesh.msh.h5"), face_count=len(rows), point_count=len(nodes),
        faces_csv_sha256=file_sha256(faces_path), nodes_csv_sha256=file_sha256(nodes_path),
        source_sha256={key:file_sha256(paths[key]) for key in ("final_case", "final_data")},
        fields={"temperature":temperature, **({"heat_flux":flux, "heat_flux_into_wall_sign":sign} if cold else {})},
        sampling="Face temperatures and flux are collocated; nodes provide actual boundary vertices")
    atomic_json(directory/"endpoint_summary.json", summary)
    return summary
