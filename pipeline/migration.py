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


def single_solve_config(cfg):
    """Select primary radiating CFD and a separate cold-wall load calibration.

    The 300 K value is reference metadata, never a prescribed CFD temperature.
    The legacy adiabatic endpoint is retired. Cold-wall flux calibrates the
    new temperature-dependent heating law; it is never a fixed thermal load.
    """
    cfg = copy.deepcopy(cfg)
    workflow = cfg.setdefault("workflow", {})
    # Endpoint-specific controls are actual inputs to the previous primary
    # solve. Preserve them when collapsing that primary into one CFD case.
    primary_overrides = copy.deepcopy(workflow.get("cold_wall_solver", {})) if workflow.get("strategy") == "two_endpoint" else {}
    workflow["strategy"] = "single_solve"
    for key in ("adiabatic_reference_iterations", "min_delta_temperature_k",
                "coordinate_tolerance_m", "cold_wall_solver", "adiabatic_solver"):
        workflow.pop(key, None)
    cfg.pop("adiabatic_reference", None)
    for key in ("_endpoint_kind", "_study_work_dir", "_cold_wall_thermal", "_study_solver"):
        cfg.pop(key, None)
    previous_thermal = cfg.get("thermal", {}) or {}
    thermal = copy.deepcopy(previous_thermal) if previous_thermal.get("mode") == "radiative_equilibrium" else {}
    thermal["mode"] = "radiative_equilibrium"
    for key, default in dict(wall_temperature_k=300.0, emissivity=.9,
                             external_radiation_temperature_k=3.0).items():
        thermal.setdefault(key, previous_thermal.get(key, default))
    cfg["thermal"] = thermal
    solver = cfg.setdefault("solver", {})
    solver.update(primary_overrides)
    # The prior generated hybrid/native profile diverged on the same mesh that
    # converged with freestream/Courant .3/first-order. Repair that complete
    # profile only; a custom CFL, scheme or initialization remains a user input.
    previous_initialization = solver.get("initialization",
        "hybrid" if solver.get("hybrid_initialize", True) else "freestream")
    failed_native_profile = (previous_initialization == "hybrid"
        and solver.get("courant_number") is None and solver.get("flow_scheme") is None)
    if failed_native_profile:
        solver.update(initialization="freestream", courant_number=.3,
                      flow_scheme="first-order-upwind")
        if solver.get("max_iterations", 1200) == 1200:
            solver["max_iterations"] = 4800
    solver.setdefault("initialization",
        ("hybrid" if solver["hybrid_initialize"] else "freestream")
        if "hybrid_initialize" in solver else "freestream")
    solver.pop("hybrid_initialize", None)
    solver.setdefault("courant_number", .3)
    solver.setdefault("flow_scheme", "first-order-upwind")
    solver.setdefault("max_iterations", 4800)
    solver.setdefault("residual_thresholds", {name: 1e-3 for name in
                                            ("continuity", "x-velocity", "y-velocity", "energy")})
    solver.setdefault("temperature_history", dict(enabled=True, sample_every_iterations=25))
    cfg.pop("thermal_export", None)
    cfg["calibration"] = _with_defaults(cfg.get("calibration", {}), dict(
        enabled=True, cold_wall_temperature_k=300.0, heat_flux_into_wall_sign=-1.0,
        temperature_field_name="temperature", heat_flux_field_name="heat-flux",
        min_delta_temperature_k=25.0, coordinate_tolerance_m=1e-9))
    cfg["calibration"]["enabled"] = True
    export = cfg.setdefault("temperature_export", {})
    export["enabled"] = True
    for key, value in dict(output_directory="temperature", static_temperature_field="temperature",
                           export_fluid_nodes=True).items():
        export.setdefault(key, value)
    # These belong to zero-flux validation, not a radiating equilibrium wall.
    export.pop("heat_flux_field", None)
    export.pop("zero_flux_tolerance_W_m2", None)
    outputs = cfg.setdefault("outputs", {})
    outputs["write_case_data"] = True
    outputs.pop("export_custom_heat_flux", None)
    outputs.pop("compute_temperature_summary", None)
    from .force_history import DEFAULTS as force_history_defaults
    aero = cfg.setdefault("aero", {})
    aero["force_history"] = _with_defaults(aero.get("force_history", {}), force_history_defaults)
    return cfg


def single_solve_state(state, old_cfg=None, new_cfg=None):
    """Keep mesh provenance and invalidate results from a different wall model."""
    state = copy.deepcopy(state)
    state.update(version=4,strategy="single_solve")
    stages = state.setdefault("stages", {})
    unchanged_primary = (old_cfg is not None and new_cfg is not None
        and old_cfg.get("workflow", {}).get("strategy") == "single_solve"
        and all(old_cfg.get(key) == new_cfg.get(key) for key in
                ("geometry", "mesh", "flight", "physics", "thermal", "solver", "temperature_export", "outputs")))
    if unchanged_primary:
        names = set()
        if old_cfg.get("calibration") != new_cfg.get("calibration"):names.update(("calibration", "thermal"))
        if old_cfg.get("aero") != new_cfg.get("aero"):names.add("aero")
    else:
        names = (set(stages) - {"mesh"}) | {"solver", "temperature_export", "aero", "calibration", "thermal"}
    for stage in names:
        stages[stage] = dict(status="stale", reason="Calibration or aerodynamic reporting configuration changed" if unchanged_primary else "Radiative-equilibrium CFD configuration changed")
    return state


def _with_defaults(existing, defaults):
    result = copy.deepcopy(existing)
    for key, default in defaults.items():
        if key not in result:
            result[key] = copy.deepcopy(default)
        elif isinstance(result[key], dict) and isinstance(default, dict):
            result[key] = _with_defaults(result[key], default)
    return result


def single_solve_thermal_config(cfg, defaults=None):
    """Select calibrated temperature-dependent heating with solved wall values."""
    previous = copy.deepcopy(cfg or {})
    result = _with_defaults(previous, defaults or {})
    if (previous.get("loading", {}) or {}).get("mode") != "aerodynamic_surface_load":
        result["loading"] = {"mode": "aerodynamic_surface_load"}
    else:
        result.setdefault("loading", {})["mode"] = "aerodynamic_surface_load"
    result["include_radiation"] = True
    # The CFD wall supplies the shared radiative environment for both anchors
    # and transient loading; old duplicate TA defaults must not override it.
    result.pop("emissivity", None)
    result.pop("radiation_temperature_k", None)
    return result


def single_solve_batch_config(cfg):
    result = copy.deepcopy(cfg)
    result.setdefault("thermal_config", "./thermal_analysis.yaml")
    result.setdefault("run_thermal", True)
    result.pop("run_adiabatic_reference", None)
    result.pop("adiabatic_required", None)
    limits = result.setdefault("timeouts", {})
    for stage, previous in (("solver", "cold_wall"), ("temperature_export", "adiabatic_export")):
        if previous in limits:
            limits.setdefault(stage, copy.deepcopy(limits[previous]))
    for stage in ("cold_wall", "cold_export", "adiabatic", "adiabatic_reference", "adiabatic_export", "thermal_bc"):
        limits.pop(stage, None)
    return result


def migrate_config(path):
    path = Path(path).resolve()
    old_bytes = path.read_bytes()
    old = yaml.safe_load(old_bytes) or {}
    new = single_solve_config(old)
    if old == new:return False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.parent / f"{path.stem}_before_single_solve_{stamp}{path.suffix}"
    shutil.copy2(path, backup)
    state_path = path.parent / "stage_state.json"
    if path.parent.name == "_batch" and state_path.is_file():
        shutil.copy2(state_path, path.parent / f"stage_state_before_single_solve_{stamp}.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state = single_solve_state(state, old, new)
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    path.write_text(yaml.safe_dump(new, sort_keys=False), encoding="utf-8")
    print(f"[migrate] Single radiative-equilibrium config: {path}; previous config: {backup}")
    return True
