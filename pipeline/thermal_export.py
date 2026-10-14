"""Export ordered body-wall face values for direct thermal heat-flux loading."""
from pathlib import Path
import csv
import math
import os
import tempfile

import numpy as np

from .utils import atomic_json


def unpack_faces(connectivity):
    """Decode Fluent's flat [node count, node ids, ...] face connectivity."""
    if isinstance(connectivity, (list, tuple)) and connectivity and np.ndim(connectivity[0]) > 0:
        return [np.asarray(row, dtype=int) for row in connectivity]
    array = np.asarray(connectivity)
    if array.ndim == 2:
        return [np.asarray(row, dtype=int) for row in array]
    if array.dtype == object:
        return [np.asarray(row, dtype=int) for row in connectivity]
    result, i = [], 0
    while i < len(array):
        count = int(array[i])
        if count < 2 or i + count >= len(array):
            raise ValueError("Invalid Fluent face connectivity")
        result.append(np.asarray(array[i + 1:i + 1 + count], dtype=int))
        i += count + 1
    return result


def order_segments(records, tolerance=1e-9):
    """Order one closed wall loop by endpoint connectivity, not centroid sorting."""
    if len(records) < 3:
        raise ValueError("Need at least three wall segments")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Topology tolerance must be finite and positive")
    adjacency, endpoints = {}, []
    for i, record in enumerate(records):
        keys = []
        for point in (record["start"], record["end"]):
            key = tuple(int(round(float(v) / tolerance)) for v in point[:2])
            adjacency.setdefault(key, []).append(i)
            keys.append(key)
        if keys[0] == keys[1]:
            raise ValueError("A wall segment collapses at the configured topology tolerance")
        endpoints.append(keys)
    if any(len(edges) != 2 for edges in adjacency.values()):
        raise ValueError("Body wall must form a single closed, non-branching loop")
    start = current = min(adjacency)
    used, ordered = set(), []
    while len(used) < len(records):
        available = [i for i in adjacency[current] if i not in used]
        if not available:
            raise ValueError("Body-wall zones contain disconnected loops")
        # A saved parallel case can enumerate faces differently. Pick the
        # neighboring vertex geometrically so both endpoints walk the loop in
        # the same direction independent of native face indices.
        i = min(available, key=lambda edge: endpoints[edge][1] if endpoints[edge][0] == current else endpoints[edge][0])
        used.add(i)
        ordered.append(records[i])
        left, right = endpoints[i]
        current = right if left == current else left
        if current == start and len(used) != len(records):
            raise ValueError("More than one closed body-wall loop was selected")
    if current != start:
        raise ValueError("Body-wall loop is not closed")
    return ordered


def export_thermal_bc_package(solver, cfg, wall_zones):
    from ansys.fluent.core.fields.field_data_interfaces import (
        ScalarFieldDataRequest, SurfaceFieldDataRequest, SurfaceDataType,
    )
    config = cfg.get("thermal_export", {})
    field_name = config.get("heat_flux_field_name", "heat-flux")
    sign = float(config.get("heat_flux_into_wall_sign", 1.0))
    if sign not in (-1.0, 1.0):
        raise ValueError("thermal_export.heat_flux_into_wall_sign must be +1 or -1")
    base = cfg["geometry"].get("body_wall_name", "body_wall")
    if not wall_zones or any(not (z == base or z.startswith(base + ".")) for z in wall_zones):
        raise ValueError("Export only named body-wall zones")
    fields = solver.fields.field_data.scalar_fields.allowed_values()
    if field_name not in fields:
        candidates = [name for name in fields if "flux" in name.casefold() or "heat" in name.casefold()]
        raise ValueError(f"Heat-flux field {field_name!r} unavailable. Inspect available fields: {candidates}")
    data = solver.fields.field_data
    geometry = data.get_field_data(SurfaceFieldDataRequest(
        surfaces=wall_zones, data_types=[SurfaceDataType.Vertices, SurfaceDataType.FacesConnectivity,
                                      SurfaceDataType.FacesCentroid], flatten_connectivity=True))
    flux = data.get_field_data(ScalarFieldDataRequest(
        surfaces=wall_zones, field_name=field_name, node_value=False, boundary_value=True))
    records = []
    for name in wall_zones:
        surface = geometry[name]
        vertices = np.asarray(surface.vertices, dtype=float)
        centroids = np.asarray(surface.face_centroids, dtype=float)
        faces = unpack_faces(surface.connectivity)
        values = np.asarray(flux[name], dtype=float).reshape(-1)
        if len(faces) != len(values) or len(centroids) != len(values):
            raise ValueError(f"Face/heat-flux count mismatch on {name}")
        if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(centroids)) or not np.all(np.isfinite(values)):
            raise ValueError("CFD wall export contains nonfinite geometry or heat flux")
        for i, face in enumerate(faces):
            if len(face) != 2:
                raise ValueError("Expected 2-D line faces; received a 3-D surface")
            if any(index < 0 or index >= len(vertices) for index in face):
                raise ValueError("Invalid wall face vertex index")
            start, end = vertices[face[0]], vertices[face[1]]
            if max(abs(start[2]), abs(end[2])) > 1e-7:
                raise ValueError("Wall data must lie in global XY")
            records.append({"zone_name": name, "face_index": i, "start": start, "end": end,
                            "centroid": centroids[i], "heat_flux_fluent_W_m2": values[i],
                            "heat_flux_into_wall_W_m2": sign * values[i]})
    ordered = order_segments(records, float(config.get("topology_tolerance_m", 1e-9)))
    rows, arc = [], 0.0
    previous = None
    for i, item in enumerate(ordered):
        x, y, z = item["centroid"]
        if previous is not None:
            arc += math.hypot(x - previous[0], y - previous[1])
        rows.append({"surface_order_index": i, "zone_name": item["zone_name"], "face_index": item["face_index"],
                     "x_m": x, "y_m": y, "z_m": z, "s_m": arc,
                     "segment_length_m": float(np.linalg.norm(item["end"] - item["start"])),
                     "heat_flux_fluent_W_m2": item["heat_flux_fluent_W_m2"],
                     "heat_flux_into_wall_W_m2": item["heat_flux_into_wall_W_m2"]})
        previous = (x, y)
    out = Path(config.get("output_directory", "thermal_bc"))
    if not out.is_absolute():
        out = Path(cfg["project"]["work_dir"]) / out
    out.mkdir(parents=True, exist_ok=True)
    output = out / "body_wall_faces.csv"
    fd, temporary = tempfile.mkstemp(prefix="body_wall_faces.", suffix=".tmp", dir=out)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()
    q = np.asarray([r["heat_flux_into_wall_W_m2"] for r in rows])
    summary = {
        "status": "complete", "body_wall_faces_csv": str(output), "face_count": len(rows),
        "source_field": field_name, "heat_flux_into_wall_sign": sign,
        "heat_flux_into_wall_W_m2": {"minimum": float(q.min()), "maximum": float(q.max()), "mean": float(q.mean())},
        "cfd_reference_wall_temperature_K": float(cfg["thermal"]["wall_temperature_k"]),
        "warning": "Verify the configured field sign on a known cold-wall case; no automatic sign flipping is performed.",
    }
    atomic_json(out / "thermal_bc_summary.json", summary)
    return summary
