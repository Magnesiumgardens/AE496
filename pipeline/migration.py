"""Explicit configuration migration, preserving geometry, flow and mesh choices."""
from datetime import datetime, timezone
from pathlib import Path
import copy
import json
import shutil

import yaml


def adiabatic_config(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["thermal"] = {"mode": "adiabatic"}
    cfg.pop("thermal_export", None)
    export = cfg.setdefault("temperature_export", {})
    for key, value in dict(enabled=True, output_directory="temperature", static_temperature_field="temperature",
                           export_fluid_nodes=True).items():
        export.setdefault(key, value)
    export.pop("heat_flux_field", None)
    export.pop("zero_flux_tolerance_W_m2", None)
    export["enabled"] = True
    outputs = cfg.setdefault("outputs", {})
    outputs.pop("export_custom_heat_flux", None)
    outputs.pop("compute_temperature_summary", None)
    outputs["write_case_data"] = True
    cfg.setdefault("solver", {}).setdefault("temperature_history", {
        "enabled": True, "sample_every_iterations": 25})
    solver = cfg["solver"]
    solver.setdefault("initialization","freestream")
    solver.setdefault("courant_number",.1)
    solver.setdefault("flow_scheme","first-order-upwind")
    solver.pop("hybrid_initialize",None)
    return cfg


def adiabatic_state(state):
    state = copy.deepcopy(state)
    stages = state.setdefault("stages", {})
    for stage in ("solver", "temperature_export", "aero"):
        stages[stage] = dict(status="stale", reason="Wall boundary changed to adiabatic zero heat flux")
    for stage in ("thermal_export", "thermal"):
        stages.pop(stage, None)
    return state


def endpoint_config(cfg):
    """Restore the documented cold-wall/warm-start baseline, without touching CAD.

    Already migrated cases retain numerical overrides. Converting a direct
    adiabatic case resets only the startup controls introduced by that workflow;
    geometry, freestream, mesh, runtime and iteration budgets remain user inputs.
    """
    cfg = copy.deepcopy(cfg)
    workflow = cfg.setdefault("workflow", {})
    already_endpoint = workflow.get("strategy") == "two_endpoint"
    workflow["strategy"] = "two_endpoint"
    old_reference = cfg.get("adiabatic_reference", {}) or {}
    workflow.setdefault("adiabatic_reference_iterations", old_reference.get("max_iterations", 600))
    workflow.setdefault("min_delta_temperature_k", 25.0)
    workflow.setdefault("coordinate_tolerance_m", 1e-9)
    thermal = cfg.get("thermal", {}) or {}
    cold_temperature = thermal.get("wall_temperature_k", 300.0) if thermal.get("mode") == "fixed_temperature" else 300.0
    if not already_endpoint:
        cfg["thermal"] = dict(mode="fixed_temperature", wall_temperature_k=cold_temperature)
    solver = cfg.setdefault("solver", {})
    solver.setdefault("temperature_history", dict(enabled=True, sample_every_iterations=25))
    if not already_endpoint:
        solver.update(initialization="hybrid", courant_number=None, flow_scheme=None,
                      residual_thresholds={name: 1e-3 for name in
                                           ("continuity", "x-velocity", "y-velocity", "energy")})
        solver.pop("hybrid_initialize", None)
    export = cfg.setdefault("thermal_export", {})
    for key, value in dict(enabled=True, output_directory="thermal_bc", temperature_field_name="temperature",
                           heat_flux_field_name="heat-flux", heat_flux_into_wall_sign=-1.0,
                           min_delta_temperature_k=25.0).items():
        export.setdefault(key, value)
    # Prior rebuilds exported the known Fluent field with the wrong default sign.
    # Retain explicit sign choices after the workflow has been migrated once.
    if not already_endpoint:
        export["enabled"] = True
        export["heat_flux_into_wall_sign"] = -1.0
    temperature = cfg.setdefault("temperature_export", {})
    for key, value in dict(enabled=True, output_directory="temperature", static_temperature_field="temperature",
                           export_fluid_nodes=True).items():
        temperature.setdefault(key, value)
    temperature.pop("heat_flux_field", None)
    temperature.pop("zero_flux_tolerance_W_m2", None)
    cfg.setdefault("outputs", {})["write_case_data"] = True
    return cfg


def endpoint_state(state):
    """Invalidate expensive downstream inputs while preserving the mesh record."""
    state = copy.deepcopy(state)
    stages = state.setdefault("stages", {})
    names = set(stages) - {"mesh"}
    names.update(("solver", "adiabatic_reference", "temperature_export", "thermal_export", "aero", "thermal"))
    for stage in names:
        stages[stage] = dict(status="stale", reason="Restored cold-wall and warm-started adiabatic CFD endpoints")
    return state


def endpoint_thermal_config(cfg, defaults=None):
    """Replace the load model while preserving exposure/material configuration."""
    result = copy.deepcopy(cfg or {})
    defaults = copy.deepcopy(defaults or {})
    for key, value in defaults.items():
        result.setdefault(key, value)
    loading = result.setdefault("loading", {})
    loading["mode"] = "convective_from_cfd"
    loading.pop("heat_transfer_coefficient_w_m2k", None)
    loading.setdefault("min_delta_temperature_k", 25.0)
    return result


def endpoint_batch_config(cfg):
    result = copy.deepcopy(cfg)
    result.setdefault("thermal_config", "./thermal_analysis.yaml")
    result.setdefault("run_thermal", True)
    # Endpoint stages are determined by workflow.strategy, not a separate flag
    # that could accidentally omit the required calibration reference.
    result.pop("run_adiabatic_reference", None)
    result.pop("adiabatic_required", None)
    limits = result.setdefault("timeouts",{})
    for stage, legacy in (("cold_wall","solver"),("adiabatic","adiabatic_reference"),
                          ("cold_export","thermal_export"),("adiabatic_export","temperature_export")):
        if legacy in limits:limits.setdefault(stage,copy.deepcopy(limits[legacy]))
    return result


def migrate_config(path):
    path = Path(path).resolve()
    old_bytes = path.read_bytes()
    old = yaml.safe_load(old_bytes) or {}
    new = endpoint_config(old)
    if old == new:return False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.parent / f"{path.stem}_before_two_endpoint_{stamp}{path.suffix}"
    shutil.copy2(path, backup)
    state_path = path.parent / "stage_state.json"
    if path.parent.name == "_batch" and state_path.is_file():
        shutil.copy2(state_path, path.parent / f"stage_state_before_two_endpoint_{stamp}.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state = endpoint_state(state)
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    path.write_text(yaml.safe_dump(new, sort_keys=False), encoding="utf-8")
    print(f"[migrate] Two-endpoint config: {path}; previous config: {backup}")
    return True
