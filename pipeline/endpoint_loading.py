"""Derive spatial convection from matched cold-wall and adiabatic CFD endpoints.

The signed cold-wall flux must point into the solid. Neither sign correction,
absolute values, clipping nor an arbitrary film coefficient is inferred here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .thermal_loading import ThermalAnalysisError, _read_rows, positive
from .temperature_export import atomic_csv, temperature_conditions
from .utils import atomic_json, file_sha256

MODEL_VERSION = "two-endpoint-convection-1"
ENDPOINTS = ("cold_wall", "adiabatic")


def convection_paths(cfg):
    directory = Path(cfg["project"]["work_dir"]) / "thermal_bc"
    return directory / "convection_nodes.csv", directory / "convection_summary.json"


def endpoint_paths(cfg, name):
    work = Path(cfg["project"]["work_dir"]) / "cfd" / name
    return work, work / "thermal_bc" / "wall_nodes.csv", work / "thermal_bc" / "endpoint_summary.json"


def source_hashes(cfg):
    work = Path(cfg["project"]["work_dir"])
    return {f"{prefix}_{kind}": file_sha256(work / "cfd" / endpoint / f"04_final.{extension}.h5")
            for endpoint, prefix in (("cold_wall", "cold"), ("adiabatic", "adiabatic"))
            for kind, extension in (("case", "cas"), ("data", "dat"))}


def _distinct_sources(cfg):
    hashes = source_hashes(cfg)
    if hashes["cold_case"] == hashes["adiabatic_case"]:
        raise ThermalAnalysisError("Cold-wall and adiabatic endpoints have identical case files; "
                                   "distinct wall conditions must be saved before convection calibration")
    return hashes


def _read_json(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):raise ValueError("Expected an object")
        return data
    except (OSError, ValueError) as exc:
        raise ThermalAnalysisError(f"Missing or invalid endpoint metadata: {path}") from exc


def _flow_conditions(conditions):
    return {key: value for key, value in conditions.items() if key != "thermal"}


def _endpoint_metadata(cfg, name):
    work, nodes, path = endpoint_paths(cfg, name)
    metadata = _read_json(path)
    current = {key: file_sha256(work / filename) for key, filename in (
        ("final_case", "04_final.cas.h5"), ("final_data", "04_final.dat.h5"))}
    if metadata.get("status") != "complete" or metadata.get("source_sha256") != current:
        raise ThermalAnalysisError(f"{name}: endpoint export does not match its current case/data pair")
    solver_summary = _read_json(work / "summary.json")
    quality = solver_summary.get("numerical_quality", {})
    if quality.get("status") != "residual_screen_passed" or quality.get("source_sha256") != current:
        raise ThermalAnalysisError(f"{name}: endpoint must pass the matching CFD residual screen before coupling")
    if solver_summary.get("source_geometry_sha256") != file_sha256(cfg["geometry"]["input_file"]):
        raise ThermalAnalysisError(f"{name}: endpoint CAD source differs from the selected geometry; regenerate CFD")
    export = cfg.get("thermal_export", {})
    expected_fields = {"temperature": export.get("temperature_field_name", "temperature")}
    if name == "cold_wall":
        sign = export.get("heat_flux_into_wall_sign")
        if isinstance(sign, bool) or sign not in (-1, 1):
            raise ThermalAnalysisError("Set an explicit thermal_export.heat_flux_into_wall_sign of -1 or +1")
        expected_fields.update(heat_flux=export.get("heat_flux_field_name", "heat-flux"), heat_flux_into_wall_sign=sign)
    if metadata.get("fields") != expected_fields:
        raise ThermalAnalysisError(f"{name}: endpoint field or heat-flux sign settings changed; re-export before coupling")
    if metadata.get("nodes_csv_sha256") != file_sha256(nodes):
        raise ThermalAnalysisError(f"{name}: exported boundary nodes were modified; re-export the endpoint")
    faces = nodes.with_name("wall_faces.csv")
    if metadata.get("faces_csv_sha256") != file_sha256(faces):
        raise ThermalAnalysisError(f"{name}: exported wall faces were modified; re-export the endpoint")
    expected = temperature_conditions(cfg)
    actual = metadata.get("conditions")
    if not isinstance(actual, dict) or _flow_conditions(actual) != _flow_conditions(expected):
        raise ThermalAnalysisError(f"{name}: flight, mesh, geometry or gas physics differ from the selected study")
    mesh_hash = file_sha256(Path(cfg["project"]["work_dir"]) / "02_mesh.msh.h5")
    if metadata.get("mesh_sha256") != mesh_hash:
        raise ThermalAnalysisError(f"{name}: endpoint mesh does not match the common study mesh")
    thermal = actual.get("thermal", {})
    if name == "adiabatic":
        if thermal != {"mode": "adiabatic"}:
            raise ThermalAnalysisError("The recovery-temperature endpoint must use a verified adiabatic wall")
        cold_work = Path(cfg["project"]["work_dir"]) / "cfd/cold_wall"
        cold_source = {key: file_sha256(cold_work / filename) for key, filename in (
            ("final_case", "04_final.cas.h5"), ("final_data", "04_final.dat.h5"))}
        if solver_summary.get("reference_source_sha256") != cold_source:
            raise ThermalAnalysisError("The adiabatic endpoint was not warm-started from the current cold-wall case/data; "
                                       "regenerate the adiabatic endpoint after changing its reference")
    elif thermal.get("mode") != "fixed_temperature":
        raise ThermalAnalysisError("The cold-wall endpoint must use a prescribed uniform wall temperature")
    elif thermal != cfg["thermal"]:
        raise ThermalAnalysisError("The cold-wall endpoint temperature differs from the selected calibration configuration")
    return nodes, metadata


def read_endpoint_nodes(path, cold=False):
    columns = ["x_m", "y_m", "z_m", "wall_temperature_K"]
    if cold:columns.append("heat_flux_into_wall_W_m2")
    rows = _read_rows(Path(path), {"surface_order_index", *columns})
    try:
        rows.sort(key=lambda row: int(row["surface_order_index"]))
        indices = [int(row["surface_order_index"]) for row in rows]
        values = np.asarray([[float(row[column]) for column in columns] for row in rows])
    except (ValueError, TypeError) as exc:
        raise ThermalAnalysisError(f"Invalid endpoint boundary-node values: {path}") from exc
    if indices != list(range(len(rows))) or len(rows) < 3 or not np.isfinite(values).all():
        raise ThermalAnalysisError("Endpoint nodes require contiguous unique order indices and finite values")
    if np.any(values[:, 3] <= 0) or np.any(np.abs(values[:, 2]) > 1e-7):
        raise ThermalAnalysisError("Endpoint temperatures must be positive kelvin on a planar XY wall")
    xy = values[:, :2]
    lengths = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
    if np.any(lengths <= 1e-12) or abs(area) <= 1e-16:
        raise ThermalAnalysisError("Endpoint boundary has duplicate nodes or does not enclose a solid")
    return values


def read_endpoint_faces(path, cold=False):
    columns = ["x_m", "y_m", "z_m", "wall_temperature_K"]
    if cold:columns.append("heat_flux_into_wall_W_m2")
    geometry = ["segment_length_m", "start_x_m", "start_y_m", "end_x_m", "end_y_m"]
    rows = _read_rows(Path(path), {"surface_order_index", "zone_name", "face_index", *columns, *geometry})
    try:
        rows.sort(key=lambda row: int(row["surface_order_index"]))
        indices = [int(row["surface_order_index"]) for row in rows]
        values = np.asarray([[float(row[column]) for column in columns] for row in rows])
        edges = np.asarray([[float(row[column]) for column in geometry] for row in rows])
        identifiers = [(row["zone_name"], int(row["face_index"])) for row in rows]
    except (ValueError, TypeError) as exc:
        raise ThermalAnalysisError(f"Invalid endpoint face values: {path}") from exc
    if (indices != list(range(len(rows))) or len(rows) < 3 or not np.isfinite(values).all()
            or not np.isfinite(edges).all() or len(set(identifiers)) != len(rows)
            or np.any(values[:, 3] <= 0) or np.any(edges[:, 0] <= 0)):
        raise ThermalAnalysisError("Endpoint wall faces require unique ordered IDs, positive lengths and finite physical values")
    return values, edges, identifiers


def match_faces_to_nodes(nodes, edges, tolerance):
    if len(nodes) != len(edges):
        raise ThermalAnalysisError("Ordered endpoint faces do not form the exported boundary-node loop")
    starts, ends = nodes[:, :2], np.roll(nodes[:, :2], -1, axis=0)
    forward = np.all(np.isclose(edges[:, 1:3], starts, rtol=0, atol=tolerance), axis=1) & np.all(
        np.isclose(edges[:, 3:5], ends, rtol=0, atol=tolerance), axis=1)
    reverse = np.all(np.isclose(edges[:, 1:3], ends, rtol=0, atol=tolerance), axis=1) & np.all(
        np.isclose(edges[:, 3:5], starts, rtol=0, atol=tolerance), axis=1)
    lengths = np.linalg.norm(ends - starts, axis=1)
    if not np.all(forward | reverse) or not np.allclose(lengths, edges[:, 0], rtol=1e-6, atol=tolerance):
        raise ThermalAnalysisError("Exported face connectivity/lengths differ from the ordered actual CFD vertices")


def face_values_to_nodes(values, lengths):
    """Adjacent-length weighted projection, preserving the closed wall loop."""
    values, lengths = np.asarray(values), np.asarray(lengths)
    return (values * lengths + np.roll(values * lengths, 1)) / (lengths + np.roll(lengths, 1))


def derive_coefficients(cold, adiabatic, min_delta_temperature_k=25, coordinate_tolerance_m=1e-9):
    """Require matching vertices and an identifiable signed film coefficient."""
    threshold = positive("min_delta_temperature_k", min_delta_temperature_k)
    tolerance = positive("coordinate_tolerance_m", coordinate_tolerance_m)
    cold, adiabatic = np.asarray(cold), np.asarray(adiabatic)
    if (cold.ndim != 2 or adiabatic.ndim != 2 or cold.shape[1] != 5 or adiabatic.shape[1] != 4
            or len(cold) != len(adiabatic) or not np.isfinite(cold).all() or not np.isfinite(adiabatic).all()):
        raise ThermalAnalysisError("Endpoint face arrays must have matching counts and finite coordinates/temperatures/flux")
    if not np.allclose(cold[:, :3], adiabatic[:, :3], rtol=0, atol=tolerance):
        raise ThermalAnalysisError("Cold and adiabatic endpoint vertices/order do not match; use the same mesh")
    difference = adiabatic[:, 3] - cold[:, 3]
    poorly_conditioned = np.flatnonzero(np.abs(difference) < threshold)
    if len(poorly_conditioned):
        raise ThermalAnalysisError(f"Cannot identify h where |T_aw - T_cold| < {threshold:g} K; "
                                   f"affected node indices: {poorly_conditioned[:10].tolist()}")
    coefficients = cold[:, 4] / difference
    invalid = np.flatnonzero(~np.isfinite(coefficients) | (coefficients < 0))
    if len(invalid):
        raise ThermalAnalysisError("Endpoint heat-flux sign is inconsistent with T_aw - T_cold; "
                                   f"negative/nonfinite h at node indices: {invalid[:10].tolist()}. "
                                   "Verify the documented Fluent flux sign; values were not flipped or clipped.")
    return coefficients


def build_convection_package(cfg):
    """Create a provenance-checked Mechanical boundary package from both endpoints."""
    try:
        sources = _distinct_sources(cfg)
        cold_path, cold_metadata = _endpoint_metadata(cfg, "cold_wall")
        adiabatic_path, adiabatic_metadata = _endpoint_metadata(cfg, "adiabatic")
        cold_nodes, adiabatic_nodes = read_endpoint_nodes(cold_path), read_endpoint_nodes(adiabatic_path)
        cold, cold_edges, cold_ids = read_endpoint_faces(cold_path.with_name("wall_faces.csv"), cold=True)
        adiabatic, adiabatic_edges, adiabatic_ids = read_endpoint_faces(adiabatic_path.with_name("wall_faces.csv"))
        settings = cfg.get("workflow", {})
        threshold = settings.get("min_delta_temperature_k", 25)
        tolerance = settings.get("coordinate_tolerance_m", 1e-9)
        coefficients_face = derive_coefficients(cold, adiabatic, threshold, tolerance)
        # Native local face indices can change when a saved parallel case is
        # repartitioned. The closed connectivity walk, matched centroids, edge
        # geometry and common mesh hash identify the physical samples.
        if [zone for zone,_ in cold_ids] != [zone for zone,_ in adiabatic_ids] or not np.allclose(cold_edges, adiabatic_edges, rtol=0, atol=tolerance):
            raise ThermalAnalysisError("Endpoint face zones or geometry differ; use the same mesh and wall zones")
        if len(cold_nodes) != len(adiabatic_nodes) or not np.allclose(cold_nodes[:, :3], adiabatic_nodes[:, :3], rtol=0, atol=tolerance):
            raise ThermalAnalysisError("Endpoint boundary vertices differ; use the same mesh")
        match_faces_to_nodes(cold_nodes, cold_edges, tolerance)
        match_faces_to_nodes(adiabatic_nodes, adiabatic_edges, tolerance)
        prescribed = positive("cold-wall temperature", cold_metadata["conditions"]["thermal"].get("wall_temperature_k"))
        if not np.allclose(cold[:, 3], prescribed, rtol=1e-6, atol=1e-5) or not np.allclose(cold_nodes[:, 3], prescribed, rtol=1e-6, atol=1e-5):
            raise ThermalAnalysisError("Cold endpoint nodal temperatures do not match its prescribed wall temperature")
        lengths = cold_edges[:, 0]
        coefficients = face_values_to_nodes(coefficients_face, lengths)
        recovery = face_values_to_nodes(adiabatic[:, 3], lengths)
        projected_flux = face_values_to_nodes(cold[:, 4], lengths)
        # Integrate the product of two linear boundary fields exactly over each
        # Mechanical line. This exposes the error introduced by face→node
        # projection instead of claiming every cold face is reproduced exactly.
        delta = recovery - prescribed
        right_h, right_delta = np.roll(coefficients, -1), np.roll(delta, -1)
        interpolated_power = float(np.sum(lengths / 6 * (
            coefficients * delta + (coefficients + right_h) * (delta + right_delta) + right_h * right_delta)))
        cold_power = float(np.sum(lengths * cold[:, 4]))
        absolute_power = float(np.sum(lengths * np.abs(cold[:, 4])))
        rows = [dict(surface_order_index=index, x_m=float(point[0]), y_m=float(point[1]), z_m=float(point[2]),
                     recovery_temperature_K=float(recovery[index]), cold_wall_temperature_K=prescribed,
                     cold_heat_flux_into_wall_W_m2=float(projected_flux[index]), heat_transfer_coefficient_W_m2K=float(coefficients[index]))
                for index, point in enumerate(cold_nodes)]
        path, summary_path = convection_paths(cfg)
        atomic_csv(path, rows)
        summary = dict(status="complete", model_version=MODEL_VERSION, conditions=temperature_conditions(cfg),
            source_sha256=sources, nodes_csv_sha256=file_sha256(path),
            mesh_sha256=cold_metadata["mesh_sha256"], point_count=len(rows),
            derivation=dict(formula="h = signed q_cold_into_wall / (T_aw - T_cold)",
                            min_delta_temperature_k=float(threshold), coordinate_tolerance_m=float(tolerance)),
            endpoint_nodes_sha256={"cold_wall":file_sha256(cold_path), "adiabatic":file_sha256(adiabatic_path)},
            endpoint_faces_sha256={"cold_wall":file_sha256(cold_path.with_name("wall_faces.csv")),
                                   "adiabatic":file_sha256(adiabatic_path.with_name("wall_faces.csv"))},
            face_matching="Closed connectivity order, centroids, endpoints and shared mesh hash; native local face indices are diagnostic only",
            diagnostics=dict(minimum_h_W_m2K=float(coefficients.min()), maximum_h_W_m2K=float(coefficients.max()),
                             minimum_absolute_delta_temperature_K=float(np.abs(adiabatic[:, 3] - cold[:, 3]).min()),
                             cold_endpoint_heat_input_per_unit_span_W_m=cold_power,
                             projected_cold_temperature_heat_input_per_unit_span_W_m=interpolated_power,
                             projection_power_error_fraction=(interpolated_power - cold_power) / absolute_power if absolute_power else 0),
            limitations=["Effective spatial h is calibrated at the fixed cold-wall temperature and held constant as the solid warms.",
                         "h is derived from collocated face temperature/flux and projected, with recovery temperature, to actual vertices using adjacent-length weights.",
                         "The cold flux column is a projected face diagnostic; piecewise-linear nodal h and recovery temperature need not reproduce every face flux exactly.",
                         "Both endpoints use the same geometry, mesh, gas model and freestream; no time-varying flight envelope is inferred.",
                         "Zero cold-wall flux with an identifiable temperature difference gives h=0; no coefficient is invented.",
                         "Residual screening does not establish mesh independence or validation of the gas model."])
        atomic_json(summary_path, summary)
        return summary
    except OSError as exc:
        raise ThermalAnalysisError(f"Endpoint calibration data are missing or unreadable: {exc}") from exc


def verify_convection_package(cfg):
    """Check source/calibration changes before feeding any load to Mechanical."""
    path, summary_path = convection_paths(cfg)
    metadata = _read_json(summary_path)
    try:
        sources = _distinct_sources(cfg)
        if (metadata.get("status") != "complete" or metadata.get("model_version") != MODEL_VERSION
                or metadata.get("conditions") != temperature_conditions(cfg)
                or metadata.get("source_sha256") != sources
                or metadata.get("nodes_csv_sha256") != file_sha256(path)):
            raise ThermalAnalysisError("Convection package is missing or stale; regenerate both endpoint exports and derived h")
        expected_mesh = file_sha256(Path(cfg["project"]["work_dir"]) / "02_mesh.msh.h5")
        if metadata.get("mesh_sha256") != expected_mesh:
            raise ThermalAnalysisError("Convection package mesh differs from the selected study mesh")
        for name in ENDPOINTS:
            nodes, _ = _endpoint_metadata(cfg, name)
            if metadata.get("endpoint_nodes_sha256", {}).get(name) != file_sha256(nodes):
                raise ThermalAnalysisError(f"{name}: endpoint nodes changed after the convection package was derived")
            if metadata.get("endpoint_faces_sha256", {}).get(name) != file_sha256(nodes.with_name("wall_faces.csv")):
                raise ThermalAnalysisError(f"{name}: endpoint faces changed after the convection package was derived")
        expected = cfg.get("workflow", {})
        for key, default in (("min_delta_temperature_k", 25), ("coordinate_tolerance_m", 1e-9)):
            if metadata.get("derivation", {}).get(key) != float(expected.get(key, default)):
                raise ThermalAnalysisError("Convection derivation settings changed; rebuild the boundary package")
    except OSError as exc:
        raise ThermalAnalysisError(f"Missing current endpoint sources: {exc}") from exc
    return path, metadata
