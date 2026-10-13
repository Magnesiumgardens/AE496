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
                           heat_flux_field="heat-flux", zero_flux_tolerance_W_m2=1.0, export_fluid_nodes=True).items():
        export.setdefault(key, value)
    export["enabled"] = True
    outputs = cfg.setdefault("outputs", {})
    outputs.pop("export_custom_heat_flux", None)
    outputs.pop("compute_temperature_summary", None)
    outputs["write_case_data"] = True
    cfg.setdefault("solver", {}).setdefault("temperature_history", {
        "enabled": True, "sample_every_iterations": 25})
    return cfg


def adiabatic_state(state):
    state = copy.deepcopy(state)
    stages = state.setdefault("stages", {})
    for stage in ("solver", "temperature_export", "aero"):
        stages[stage] = dict(status="stale", reason="Wall boundary changed to adiabatic zero heat flux")
    for stage in ("thermal_export", "thermal"):
        stages.pop(stage, None)
    return state


def migrate_config(path):
    path = Path(path).resolve()
    old_bytes = path.read_bytes()
    old = yaml.safe_load(old_bytes) or {}
    new = adiabatic_config(old)
    if old == new:return False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.parent / f"{path.stem}_before_adiabatic_{stamp}{path.suffix}"
    shutil.copy2(path, backup)
    state_path = path.parent / "stage_state.json"
    if path.parent.name == "_batch" and state_path.is_file():
        shutil.copy2(state_path, path.parent / f"stage_state_before_adiabatic_{stamp}.json")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state = adiabatic_state(state)
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    path.write_text(yaml.safe_dump(new, sort_keys=False), encoding="utf-8")
    print(f"[migrate] Adiabatic config: {path}; previous config: {backup}")
    return True
