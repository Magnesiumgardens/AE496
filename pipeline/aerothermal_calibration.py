"""Match cold-wall flux with radiative-equilibrium temperatures on one CFD mesh.

This is a two-state *load calibration*, not the legacy adiabatic endpoint
workflow. The primary solution remains the root radiative-equilibrium case.
An affine gas-side heating law is identified at collocated wall faces, then
its conductance and source are projected separately to actual wall vertices.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .endpoint_loading import match_faces_to_nodes, read_endpoint_faces, read_endpoint_nodes
from .temperature_export import atomic_csv, temperature_conditions
from .thermal_loading import ThermalAnalysisError, positive, _read_rows
from .thermal_scope import active_segments, active_nodes, scoped_face_values_to_nodes
from .utils import atomic_json, file_sha256

MODEL_VERSION = "re-cold-affine-heating-2"
STEFAN_BOLTZMANN = 5.670374419e-8


def calibration_settings(cfg):
    values = cfg.get("calibration", {}) or {}
    sign = values.get("heat_flux_into_wall_sign", -1)
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ThermalAnalysisError("calibration.heat_flux_into_wall_sign must explicitly be -1 or +1")
    thermal = cfg.get("thermal", {})
    emissivity = float(thermal.get("emissivity", .9))
    if not np.isfinite(emissivity) or not 0 < emissivity <= 1:
        raise ThermalAnalysisError("Radiative calibration requires 0 < thermal.emissivity <= 1")
    exclude_rear = values.get("exclude_rear_face", False)
    if not isinstance(exclude_rear, bool):
        raise ThermalAnalysisError("calibration.exclude_rear_face must be true or false")
    try:
        direction = np.asarray(cfg.get("reporting", {}).get("nose_direction", [1., 0.]), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ThermalAnalysisError("reporting.nose_direction must be a finite nonzero 2-D vector") from exc
    if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ThermalAnalysisError("reporting.nose_direction must be a finite nonzero 2-D vector")
    direction = direction / np.linalg.norm(direction)
    return dict(cold_wall_temperature_k=positive("calibration.cold_wall_temperature_k", values.get("cold_wall_temperature_k", 300)),
                heat_flux_into_wall_sign=sign,
                temperature_field_name=str(values.get("temperature_field_name", "temperature")),
                heat_flux_field_name=str(values.get("heat_flux_field_name", "heat-flux")),
                min_delta_temperature_k=positive("calibration.min_delta_temperature_k", values.get("min_delta_temperature_k", 25)),
                coordinate_tolerance_m=positive("calibration.coordinate_tolerance_m", values.get("coordinate_tolerance_m", 1e-9)),
                exclude_rear_face=exclude_rear,
                rear_plane_tolerance_fraction=positive("calibration.rear_plane_tolerance_fraction", values.get("rear_plane_tolerance_fraction", 1e-7)),
                nose_direction=direction.tolist(),
                emissivity=emissivity,
                external_radiation_temperature_K=positive("thermal.external_radiation_temperature_k", thermal.get("external_radiation_temperature_k", 3)))


def calibration_paths(cfg):
    root = Path(cfg.get("_study_work_dir", cfg["project"]["work_dir"])).resolve()
    cold = root / "cfd" / "cold_calibration"
    directory = root / "thermal_bc"
    return dict(root=root, re_work=root, cold_work=cold,
                re_faces=directory / "wall_faces.csv", re_nodes=directory / "wall_nodes.csv",
                re_summary=directory / "endpoint_summary.json",
                cold_faces=cold / "thermal_bc/wall_faces.csv", cold_nodes=cold / "thermal_bc/wall_nodes.csv",
                cold_summary=cold / "thermal_bc/endpoint_summary.json",
                faces=directory / "surface_load_faces.csv", nodes=directory / "surface_load_nodes.csv",
                summary=directory / "surface_load_summary.json")


def calibration_cfg(cfg):
    """Internal cold-wall calibration using the unchanged root study mesh."""
    settings = calibration_settings(cfg)
    paths = calibration_paths(cfg)
    result = copy.deepcopy(cfg)
    result.update(_calibration_kind="cold_wall", _study_work_dir=str(paths["root"]))
    result.pop("_endpoint_kind", None)
    result["project"]["work_dir"] = str(paths["cold_work"])
    result["thermal"] = dict(mode="fixed_temperature", wall_temperature_k=settings["cold_wall_temperature_k"])
    result.setdefault("workflow", {})["strategy"] = "single_solve"
    result.setdefault("solver", {}).setdefault("temperature_history", {})["enabled"] = False
    result["thermal_export"] = {key: settings[key] for key in (
        "temperature_field_name", "heat_flux_field_name", "heat_flux_into_wall_sign")}
    result["workflow"]["coordinate_tolerance_m"] = settings["coordinate_tolerance_m"]
    result.pop("adiabatic_reference", None)
    return result


def _re_export_cfg(cfg):
    result = copy.deepcopy(cfg)
    settings = calibration_settings(cfg)
    result.setdefault("thermal_export", {})["temperature_field_name"] = settings["temperature_field_name"]
    result.setdefault("workflow", {})["coordinate_tolerance_m"] = settings["coordinate_tolerance_m"]
    return result


def export_re_calibration(solver, cfg, zones):
    """Save collocated RE face temperatures and actual boundary vertices.

    A generic Fluent ``heat-flux`` field is deliberately not relabeled as the
    separate convective contribution on a Radiation wall. Its convective value
    is derived later from the verified external-radiation equilibrium balance.
    """
    if cfg.get("thermal", {}).get("mode") != "radiative_equilibrium":
        raise ThermalAnalysisError("The primary calibration state must be radiative_equilibrium")
    from .endpoint_export import export_endpoint
    return export_endpoint(solver, _re_export_cfg(cfg), zones)


def export_cold_calibration(solver, cfg, zones):
    """Export signed native cold-wall face flux; no direct solid load is made."""
    from .endpoint_export import export_endpoint
    return export_endpoint(solver, calibration_cfg(cfg), zones)


def _read_json(path):
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Expected object")
        return result
    except (OSError, ValueError) as exc:
        raise ThermalAnalysisError(f"Missing or invalid aerothermal calibration metadata: {path}") from exc


def source_hashes(cfg):
    paths = calibration_paths(cfg)
    return {f"{state}_{kind}": file_sha256(paths[f"{state}_work"] / filename)
            for state in ("re", "cold")
            for kind, filename in (("case", "04_final.cas.h5"), ("data", "04_final.dat.h5"))}


def _state_metadata(cfg, state):
    paths = calibration_paths(cfg)
    effective = calibration_cfg(cfg) if state == "cold" else _re_export_cfg(cfg)
    work = paths[f"{state}_work"]
    metadata = _read_json(paths[f"{state}_summary"])
    current = {key: file_sha256(work / filename) for key, filename in (
        ("final_case", "04_final.cas.h5"), ("final_data", "04_final.dat.h5"))}
    if metadata.get("status") != "complete" or metadata.get("source_sha256") != current:
        raise ThermalAnalysisError(f"{state}: calibration export does not match the saved CFD case/data")
    solved = _read_json(work / "summary.json")
    quality = solved.get("numerical_quality", {})
    if quality.get("status") != "residual_screen_passed" or quality.get("source_sha256") != current:
        raise ThermalAnalysisError(f"{state}: calibration requires the matching passed CFD residual screen")
    geometry_hash = file_sha256(cfg["geometry"]["input_file"])
    mesh_hash = file_sha256(paths["root"] / "02_mesh.msh.h5")
    if solved.get("source_geometry_sha256") != geometry_hash:
        raise ThermalAnalysisError(f"{state}: calibration CAD source differs from the selected geometry")
    if solved.get("source_mesh_sha256") != mesh_hash or metadata.get("mesh_sha256") != mesh_hash:
        raise ThermalAnalysisError(f"{state}: calibration must use the unchanged common study mesh")
    if metadata.get("conditions") != temperature_conditions(effective):
        raise ThermalAnalysisError(f"{state}: calibration flight, gas physics, geometry or wall conditions changed")
    settings = calibration_settings(cfg)
    fields = dict(temperature=settings["temperature_field_name"])
    if state == "cold":
        fields.update(heat_flux=settings["heat_flux_field_name"], heat_flux_into_wall_sign=settings["heat_flux_into_wall_sign"])
    if metadata.get("fields") != fields:
        raise ThermalAnalysisError(f"{state}: calibration field names or signed flux convention changed")
    for key in ("faces", "nodes"):
        if metadata.get(f"{key}_csv_sha256") != file_sha256(paths[f"{state}_{key}"]):
            raise ThermalAnalysisError(f"{state}: exported calibration {key} were changed after export")
    return metadata


def verify_calibration_export(cfg, state):
    if state not in ("re", "cold"):
        raise ValueError(f"Unknown calibration state: {state}")
    try:
        return _state_metadata(cfg, state)
    except OSError as exc:
        raise ThermalAnalysisError(f"Calibration source files are missing: {exc}") from exc


def derive_affine_heating(cold_temperature, equilibrium_temperature, cold_flux, emissivity,
                          radiation_temperature, min_delta_temperature_k=25):
    """Fit q_conv(T)=source-H*T at matched CFD faces, without clipping.

    Both anchors are gas-side heat *into* the solid. At RE the anchor follows
    q_eq=epsilon*sigma*(T_eq**4-T_rad**4), because the wall has no storage or
    interior solid-conduction model. This is not a measured adiabatic T_aw.
    """
    tc, teq, qc = np.broadcast_arrays(np.asarray(cold_temperature, dtype=float),
                                    np.asarray(equilibrium_temperature, dtype=float),
                                    np.asarray(cold_flux, dtype=float))
    if not all(np.isfinite(array).all() for array in (tc, teq, qc)) or np.any(tc <= 0) or np.any(teq <= 0):
        raise ThermalAnalysisError("Calibration temperatures/flux must be finite, with positive kelvin temperatures")
    eps = float(emissivity)
    trad = positive("external radiation temperature", radiation_temperature)
    if not np.isfinite(eps) or not 0 < eps <= 1:
        raise ThermalAnalysisError("Radiative calibration requires emissivity in (0, 1]")
    delta = teq - tc
    minimum = positive("min_delta_temperature_k", min_delta_temperature_k)
    if np.any(delta < minimum):
        raise ThermalAnalysisError(f"Radiative-equilibrium calibration requires T_eq - T_cold >= {minimum:g} K at every face")
    qeq = eps * STEFAN_BOLTZMANN * (teq**4 - trad**4)
    conductance = (qc - qeq) / delta
    source = qc + conductance * tc
    if (not np.isfinite(qeq).all() or not np.isfinite(conductance).all()
            or not np.isfinite(source).all() or np.any(conductance <= 0)):
        raise ThermalAnalysisError("Cold and RE heat-flux anchors do not identify positive aerodynamic conductance; verify heat-flux sign, wall states and convergence. No sign flip or clipping was applied.")
    return conductance, source, qeq


def _match_states(re_nodes, re_faces, re_edges, re_ids, cold_nodes, cold_faces, cold_edges, cold_ids, tolerance):
    """Match physical edges, not native face numbers or parallel partition order."""
    match_faces_to_nodes(re_nodes, re_edges, tolerance)
    match_faces_to_nodes(cold_nodes, cold_edges, tolerance)
    if len(re_faces) != len(cold_faces) or len(re_nodes) != len(cold_nodes):
        raise ThermalAnalysisError("RE and cold calibration surface counts differ")
    permutation = []
    for index, edge in enumerate(re_edges):
        forward = np.all(np.isclose(cold_edges[:, 1:3], edge[1:3], rtol=0, atol=tolerance), axis=1) & np.all(
            np.isclose(cold_edges[:, 3:5], edge[3:5], rtol=0, atol=tolerance), axis=1)
        reverse = np.all(np.isclose(cold_edges[:, 1:3], edge[3:5], rtol=0, atol=tolerance), axis=1) & np.all(
            np.isclose(cold_edges[:, 3:5], edge[1:3], rtol=0, atol=tolerance), axis=1)
        same_center = np.all(np.isclose(cold_faces[:, :3], re_faces[index, :3], rtol=0, atol=tolerance), axis=1)
        same_zone = np.array([zone == re_ids[index][0] for zone, _ in cold_ids])
        match = np.flatnonzero((forward | reverse) & same_center & same_zone & np.isclose(
            cold_edges[:, 0], edge[0], rtol=1e-6, atol=tolerance))
        if len(match) != 1:
            raise ThermalAnalysisError("RE and cold calibration face zones, centroids or endpoint geometry do not match uniquely")
        permutation.append(int(match[0]))
    if len(set(permutation)) != len(permutation):
        raise ThermalAnalysisError("Calibration face matching is not one-to-one")
    # Every native node must also identify one vertex of the same closed loop.
    for point in re_nodes[:, :3]:
        matches = np.all(np.isclose(cold_nodes[:, :3], point, rtol=0, atol=tolerance), axis=1)
        if np.count_nonzero(matches) != 1:
            raise ThermalAnalysisError("RE and cold calibration actual boundary vertices differ")
    return np.asarray(permutation, dtype=int)


def _matched_calibration_inputs(cfg):
    """Read and validate the same sources for calibration and diagnostics."""
    try:
        if cfg.get("thermal", {}).get("mode") != "radiative_equilibrium":
            raise ThermalAnalysisError("Aerodynamic surface-load calibration requires the primary radiative-equilibrium CFD state")
        settings = calibration_settings(cfg)
        paths = calibration_paths(cfg)
        sources = source_hashes(cfg)
        if sources["re_case"] == sources["cold_case"]:
            raise ThermalAnalysisError("RE and cold calibration have identical case files despite different wall states")
        re_metadata, cold_metadata = _state_metadata(cfg, "re"), _state_metadata(cfg, "cold")
        re_nodes, cold_nodes = read_endpoint_nodes(paths["re_nodes"]), read_endpoint_nodes(paths["cold_nodes"])
        re_faces, re_edges, re_ids = read_endpoint_faces(paths["re_faces"])
        cold_faces, cold_edges, cold_ids = read_endpoint_faces(paths["cold_faces"], cold=True)
        order = _match_states(re_nodes, re_faces, re_edges, re_ids, cold_nodes, cold_faces, cold_edges, cold_ids,
                              settings["coordinate_tolerance_m"])
        cold_faces = cold_faces[order]
        tc = settings["cold_wall_temperature_k"]
        if not np.allclose(cold_faces[:, 3], tc, rtol=1e-6, atol=1e-5) or not np.allclose(cold_nodes[:, 3], tc, rtol=1e-6, atol=1e-5):
            raise ThermalAnalysisError("Cold calibration field temperatures do not match its prescribed wall temperature")
        return (settings, paths, sources, re_metadata, re_nodes, re_faces, re_edges, re_ids, cold_faces, cold_ids, order)
    except OSError as exc:
        raise ThermalAnalysisError(f"Aerothermal calibration files are missing or unreadable: {exc}") from exc


def _anchor_diagnostics(cfg, matched, *, write=False):
    settings, paths, sources, _, re_nodes, re_faces, edges, re_ids, cold_faces, cold_ids, order = matched
    active = _active_calibration_segments(settings, re_nodes)
    tc, teq, qc = cold_faces[:, 3], re_faces[:, 3], cold_faces[:, 4]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        qeq = settings["emissivity"] * STEFAN_BOLTZMANN * (teq**4 - settings["external_radiation_temperature_K"]**4)
        delta, gap = teq - tc, qc - qeq
        h = gap / delta
        source = qc + h * tc
    def finite(value):
        return float(value) if np.isfinite(value) else None
    rows = []
    for index, point in enumerate(re_faces):
        if delta[index] < settings["min_delta_temperature_k"]:
            status = "insufficient_temperature_separation"
        elif not all(np.isfinite(array[index]) for array in (qeq, h, source)):
            status = "nonfinite_calibration"
        elif h[index] <= 0:
            status = "nonpositive_conductance"
        else:
            status = "passed"
        scale = max(abs(qc[index]), abs(qeq[index]))
        relative_gap = finite(gap[index] / scale) if np.isfinite(scale) and scale > 0 else (0. if scale == 0 else None)
        rows.append(dict(surface_order_index=index, zone_name=re_ids[index][0],
            re_face_index=re_ids[index][1], cold_face_index=cold_ids[order[index]][1],
            x_m=float(point[0]), y_m=float(point[1]), segment_length_m=float(edges[index, 0]),
            cold_wall_temperature_K=float(tc[index]), equilibrium_temperature_K=float(teq[index]),
            cold_heat_flux_into_wall_W_m2=float(qc[index]), equilibrium_convective_heat_flux_W_m2=finite(qeq[index]),
            temperature_separation_K=float(delta[index]), cold_minus_equilibrium_heat_flux_W_m2=finite(gap[index]),
            relative_anchor_gap=relative_gap,
            candidate_conductance_W_m2K=finite(h[index]), thermal_active=int(active[index]), status=status))
    failed = [row for row in rows if row["thermal_active"] and row["status"] != "passed"]
    excluded = [row for row in rows if not row["thermal_active"]]
    worst = sorted(failed, key=lambda row: row["candidate_conductance_W_m2K"]
                   if row["candidate_conductance_W_m2K"] is not None else -np.inf)[:10]
    finite_h = h[np.isfinite(h) & active]
    summary = dict(status="failed" if failed else "passed", case=cfg["project"]["name"],
        source_sha256=sources, calibration_settings=settings, face_count=len(rows), invalid_face_count=len(failed),
        active_face_count=int(np.count_nonzero(active)), excluded_face_count=len(excluded),
        excluded_invalid_face_count=sum(row["status"] != "passed" for row in excluded),
        nonpositive_conductance_face_count=sum(row["status"] == "nonpositive_conductance" for row in failed),
        insufficient_temperature_separation_face_count=sum(row["status"] == "insufficient_temperature_separation" for row in failed),
        nonpositive_cold_heat_flux_face_count=int(np.count_nonzero((qc <= 0) & active)),
        failed_boundary_length_fraction=float(sum(row["segment_length_m"] for row in failed) / np.sum(edges[:, 0])),
        minimum_conductance_W_m2K=finite(finite_h.min()) if len(finite_h) else None,
        maximum_conductance_W_m2K=finite(finite_h.max()) if len(finite_h) else None,
        minimum_cold_heat_flux_into_wall_W_m2=finite(qc[active].min()), maximum_cold_heat_flux_into_wall_W_m2=finite(qc[active].max()),
        minimum_equilibrium_temperature_K=finite(teq[active].min()), maximum_equilibrium_temperature_K=finite(teq[active].max()),
        worst_faces=worst, excluded_faces=excluded,
        interpretation="A passed CFD residual screen does not guarantee a positive local two-state heating slope. No coefficient is changed by this diagnosis. Explicitly excluded rear faces are retained as diagnostics but do not enter thermal calibration.",
        next_step="Inspect matched face temperatures, fluxes and positions before changing signs, wall states, numerics or the heating model.")
    if write:
        directory = paths["root"] / "thermal_bc"
        summary.update(diagnostics_csv=str(directory / "calibration_diagnostics.csv"),
                       diagnostics_json=str(directory / "calibration_diagnostics.json"))
        atomic_csv(summary["diagnostics_csv"], rows)
        atomic_json(summary["diagnostics_json"], summary)
    return summary


def _active_calibration_segments(settings, nodes):
    return active_segments(nodes[:, :2], exclude_rear_face=settings["exclude_rear_face"],
        nose_direction=settings["nose_direction"], tolerance_fraction=settings["rear_plane_tolerance_fraction"])


def diagnose_calibration(cfg, *, write=False):
    """Inspect existing matched exports without launching a solver or fitting a usable load.

    The default is read-only. The diagnostic CSV/JSON are never accepted as a
    surface-load package, and negative slopes retain their original sign.
    """
    return _anchor_diagnostics(cfg, _matched_calibration_inputs(cfg), write=write)


def build_surface_package(cfg):
    """Build the provenance-checked, temperature-dependent solid heating input."""
    matched = _matched_calibration_inputs(cfg)
    settings, paths, sources, re_metadata, re_nodes, re_faces, re_edges, re_ids, cold_faces, cold_ids, order = matched
    tc = settings["cold_wall_temperature_k"]
    active = _active_calibration_segments(settings, re_nodes)
    loaded_nodes = active_nodes(active)
    try:
        diagnostics = _anchor_diagnostics(cfg, matched, write=True)
        try:
            selected_h, selected_source, selected_qeq = derive_affine_heating(cold_faces[active, 3], re_faces[active, 3], cold_faces[active, 4],
                settings["emissivity"], settings["external_radiation_temperature_K"], settings["min_delta_temperature_k"])
        except ThermalAnalysisError as exc:
            raise ThermalAnalysisError(f"{exc} Rejected {diagnostics['invalid_face_count']} of {diagnostics['face_count']} matched faces. "
                f"See {diagnostics['diagnostics_csv']} and {diagnostics['diagnostics_json']}. "
                "Inspect saved results with study.py calibration-status --config <case.yaml>; no additional CFD is needed for that diagnosis.") from exc
        # Inactive coefficients are explicit uncalibrated placeholders. Rear
        # geometry remains present; no valid heating law is invented there.
        h, source, qeq = (np.zeros(len(active)) for _ in range(3))
        h[active], source[active], qeq[active] = selected_h, selected_source, selected_qeq
        lengths = re_edges[:, 0]
        nodal_h = scoped_face_values_to_nodes(h, lengths, active)
        nodal_source = scoped_face_values_to_nodes(source, lengths, active)
        nodal_qcold = scoped_face_values_to_nodes(cold_faces[:, 4], lengths, active)
        nodal_qeq = scoped_face_values_to_nodes(qeq, lengths, active)
        driving = np.divide(nodal_source, nodal_h, out=np.zeros_like(nodal_source), where=loaded_nodes)
        right_h, right_driving = np.roll(nodal_h, -1), np.roll(driving, -1)
        cold_power = float(np.sum(lengths[active] * cold_faces[active, 4]))
        absolute_cold_power = float(np.sum(lengths[active] * np.abs(cold_faces[active, 4])))
        native_cold_power = float(np.sum((lengths / 6 * (
            nodal_h * driving + (nodal_h + right_h) * (driving + right_driving)
            + right_h * right_driving) - lengths * tc * (nodal_h + right_h) / 2)[active]))
        face_rows = []
        for index, point in enumerate(re_faces):
            edge = re_edges[index]
            face_rows.append(dict(surface_order_index=index, zone_name=re_ids[index][0],
                re_face_index=re_ids[index][1], cold_face_index=cold_ids[order[index]][1],
                x_m=float(point[0]), y_m=float(point[1]), z_m=float(point[2]), segment_length_m=float(edge[0]),
                start_x_m=float(edge[1]), start_y_m=float(edge[2]), end_x_m=float(edge[3]), end_y_m=float(edge[4]),
                thermal_active=int(active[index]),
                equilibrium_temperature_K=float(point[3]), cold_wall_temperature_K=tc,
                cold_heat_flux_into_wall_W_m2=float(cold_faces[index, 4]), equilibrium_convective_heat_flux_W_m2=float(qeq[index]),
                aerodynamic_conductance_W_m2K=float(h[index]), aerodynamic_source_W_m2=float(source[index]),
                driving_temperature_K=float(source[index] / h[index]) if active[index] else 0.))
        node_rows = [dict(surface_order_index=index, x_m=float(point[0]), y_m=float(point[1]), z_m=float(point[2]),
            equilibrium_temperature_K=float(point[3]), cold_wall_temperature_K=tc,
            cold_heat_flux_into_wall_W_m2=float(nodal_qcold[index]), equilibrium_convective_heat_flux_W_m2=float(nodal_qeq[index]),
            aerodynamic_conductance_W_m2K=float(nodal_h[index]), aerodynamic_source_W_m2=float(nodal_source[index]),
            driving_temperature_K=float(driving[index]), thermal_active=int(loaded_nodes[index]),
            thermal_segment_active=int(active[index])) for index, point in enumerate(re_nodes)]
        atomic_csv(paths["faces"], face_rows)
        atomic_csv(paths["nodes"], node_rows)
        # The face anchors are exact. Projection smooths H and the source; actual
        # native RE nodal temperatures are diagnostics, not prescribed values.
        nodal_re_residual = nodal_source[loaded_nodes] - nodal_h[loaded_nodes] * re_nodes[loaded_nodes, 3] - settings["emissivity"] * STEFAN_BOLTZMANN * (
            re_nodes[loaded_nodes, 3]**4 - settings["external_radiation_temperature_K"]**4)
        summary = dict(status="complete", model_version=MODEL_VERSION, loading_mode="aerodynamic_surface_load",
            conditions=temperature_conditions(cfg), source_sha256=sources,
            source_geometry_sha256=file_sha256(cfg["geometry"]["input_file"]), mesh_sha256=re_metadata["mesh_sha256"],
            faces_csv_sha256=file_sha256(paths["faces"]), nodes_csv_sha256=file_sha256(paths["nodes"]),
            face_count=len(face_rows), point_count=len(node_rows), calibration_settings=settings,
            thermal_scope=dict(rear_boundary="insulated" if settings["exclude_rear_face"] else "aerodynamically_loaded",
                active_face_count=int(np.count_nonzero(active)), excluded_face_count=int(np.count_nonzero(~active)),
                active_boundary_length_m=float(np.sum(lengths[active])), excluded_boundary_length_m=float(np.sum(lengths[~active])),
                thermal_segment_active=active.astype(int).tolist(), thermal_active=loaded_nodes.astype(int).tolist(),
                geometry="Complete closed CFD/solid boundary retained", aerodynamic_force_scope="Complete CFD body wall unchanged"),
            radiation=dict(emissivity=settings["emissivity"], external_radiation_temperature_K=settings["external_radiation_temperature_K"],
                           stefan_boltzmann_W_m2K4=STEFAN_BOLTZMANN),
            derivation=dict(formula="q_conv(T_w) = aerodynamic_source - aerodynamic_conductance * T_w",
                conductance="(q_cold - q_eq) / (T_eq - T_cold)",
                re_convective_flux_source="radiative_equilibrium_balance",
                equilibrium_flux="epsilon * sigma * (T_eq**4 - T_rad**4)",
                min_delta_temperature_k=settings["min_delta_temperature_k"], coordinate_tolerance_m=settings["coordinate_tolerance_m"]),
            endpoint_nodes_sha256={state:file_sha256(paths[f"{state}_nodes"]) for state in ("re", "cold")},
            endpoint_faces_sha256={state:file_sha256(paths[f"{state}_faces"]) for state in ("re", "cold")},
            endpoint_metadata_sha256={state:file_sha256(paths[f"{state}_summary"]) for state in ("re", "cold")},
            face_matching="Closed loops, zone names, centroids, endpoints and common mesh; native local face indices are diagnostic only",
            projection="Adjacent active-face-length-weighted conductance and source projected separately; inactive rear faces never contribute to corner values. Driving temperature is projected source / projected conductance on loaded nodes only.",
            diagnostics=dict(minimum_conductance_W_m2K=float(h[active].min()), maximum_conductance_W_m2K=float(h[active].max()),
                cold_face_heat_input_per_unit_span_W_m=cold_power,
                native_linear_conductance_driving_temperature_cold_power_W_m=native_cold_power,
                native_linear_conductance_driving_temperature_cold_power_error_fraction=(native_cold_power - cold_power) / absolute_cold_power if absolute_cold_power else None,
                equilibrium_face_heat_input_per_unit_span_W_m=float(np.sum(lengths * qeq)),
                maximum_projected_RE_nodal_net_flux_residual_W_m2=float(np.max(np.abs(nodal_re_residual)))),
            limitations=["Affine gas heating is calibrated between two steady wall states at one fixed flight condition; the fitted conductance is not independently temperature-dependent.",
                "RE convective flux is inferred from its external radiation equilibrium balance, not asserted to be a separately measured Fluent convective field.",
                "The cold flux is a calibration anchor, never a constant transient thermal load.",
                "Gas heating is evaluated at the evolving solid wall temperature; there is no transient CFD feedback or Mach/AoA sweep.",
                "Face calibration anchors are exact; separate conductance/source projection onto boundary vertices smooths local fields and need not reproduce native RE nodal equilibrium exactly.",
                "MAPDL CONV interpolates conductance and driving temperature separately; their product differs from a linearly interpolated source between vertices, with the cold-load integral error reported explicitly.",
                "The local affine law approximates two complete steady wall-state solutions; it is not an independently measured local derivative or a conjugate-flow response to arbitrary wall-temperature patterns.",
                "Values outside the two wall-temperature states are affine extrapolations; no clipping or gas chemistry correction is applied.",
                *( ["The excluded rear closure is an insulated solid cut: zero applied convection/radiation and no modeled conduction into an attached vehicle; the existing full-boundary CFD and wake remain unchanged."] if settings["exclude_rear_face"] else []),
                "Residual screening and shared-source checks do not establish mesh independence or experimental validity."])
        atomic_json(paths["summary"], summary)
        return summary
    except OSError as exc:
        raise ThermalAnalysisError(f"Aerothermal calibration files are missing or unreadable: {exc}") from exc


def verify_surface_package(cfg):
    """Reject stale CFD, fields, geometry, material radiation anchors or CSVs."""
    paths = calibration_paths(cfg)
    metadata = _read_json(paths["summary"])
    try:
        if (metadata.get("status") != "complete" or metadata.get("model_version") != MODEL_VERSION
                or metadata.get("conditions") != temperature_conditions(cfg)
                or metadata.get("source_sha256") != source_hashes(cfg)
                or metadata.get("calibration_settings") != calibration_settings(cfg)):
            raise ThermalAnalysisError("Aerodynamic surface-load package is missing or stale; rebuild calibration")
        if metadata.get("source_geometry_sha256") != file_sha256(cfg["geometry"]["input_file"]):
            raise ThermalAnalysisError("Aerothermal calibration CAD source differs from the selected geometry")
        if metadata.get("mesh_sha256") != file_sha256(paths["root"] / "02_mesh.msh.h5"):
            raise ThermalAnalysisError("Aerothermal calibration mesh differs from the selected common mesh")
        for field in ("faces", "nodes"):
            if metadata.get(f"{field}_csv_sha256") != file_sha256(paths[field]):
                raise ThermalAnalysisError(f"Aerothermal surface-load {field} CSV changed after calibration")
        for state in ("re", "cold"):
            _state_metadata(cfg, state)
            for field in ("faces", "nodes"):
                if metadata.get(f"endpoint_{field}_sha256", {}).get(state) != file_sha256(paths[f"{state}_{field}"]):
                    raise ThermalAnalysisError(f"{state}: endpoint {field} changed after surface-load calibration")
            if metadata.get("endpoint_metadata_sha256", {}).get(state) != file_sha256(paths[f"{state}_summary"]):
                raise ThermalAnalysisError(f"{state}: endpoint metadata changed after surface-load calibration")
        settings = calibration_settings(cfg)
        vertices = read_endpoint_nodes(paths["re_nodes"])
        selected = _active_calibration_segments(settings, vertices)
        selected_nodes = active_nodes(selected)
        scope = metadata.get("thermal_scope", {})
        if (scope.get("thermal_segment_active") != selected.astype(int).tolist()
                or scope.get("thermal_active") != selected_nodes.astype(int).tolist()
                or scope.get("rear_boundary") != ("insulated" if settings["exclude_rear_face"] else "aerodynamically_loaded")):
            raise ThermalAnalysisError("Aerothermal package thermal scope differs from the configured geometric rear boundary")
        for key, expected in (("faces", selected), ("nodes", selected_nodes)):
            required = {"surface_order_index", "thermal_active"}
            if key == "nodes":required.add("thermal_segment_active")
            rows = _read_rows(paths[key], required)
            try:
                rows.sort(key=lambda row: int(row["surface_order_index"]))
                actual = [int(row["thermal_active"]) for row in rows]
                indices = [int(row["surface_order_index"]) for row in rows]
                outgoing = [int(row["thermal_segment_active"]) for row in rows] if key == "nodes" else selected.astype(int).tolist()
            except (ValueError, TypeError) as exc:
                raise ThermalAnalysisError("Aerothermal package has invalid thermal scope flags") from exc
            if (indices != list(range(len(selected))) or actual != expected.astype(int).tolist()
                    or outgoing != selected.astype(int).tolist()):
                raise ThermalAnalysisError("Aerothermal package thermal scope flags do not match its full closed geometry")
    except OSError as exc:
        raise ThermalAnalysisError(f"Current aerothermal calibration sources are missing: {exc}") from exc
    return paths["nodes"], metadata
