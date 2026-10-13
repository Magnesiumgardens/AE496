"""Load one authoritative case configuration with paths relative to its YAML."""
from pathlib import Path
import math

import yaml


class ConfigError(ValueError):
    pass


def _required(cfg, dotted):
    value = cfg
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ConfigError(f"Missing required config value: {dotted}")
        value = value[key]
    return value


def positive(name, value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ConfigError(f"{name} must be a finite positive number")
    return number


def validate_config(cfg):
    for name in ("project.name", "project.work_dir", "geometry.input_file",
                 "flight.flow_direction", "thermal.mode",
                 "physics.air", "solver.max_iterations"):
        _required(cfg, name)
    for name in ("flight.mach", "flight.static_pressure_pa", "flight.static_temperature_k",
                 "mesh.global_min_size", "mesh.global_max_size",
                 "mesh.body_min_size", "mesh.body_max_size"):
        positive(name, _required(cfg, name))
    for prefix in ("global", "body"):
        if float(cfg["mesh"][prefix + "_min_size"]) > float(cfg["mesh"][prefix + "_max_size"]):
            raise ConfigError(f"mesh.{prefix}_min_size cannot exceed its maximum")
    direction = cfg["flight"]["flow_direction"]
    if not isinstance(direction, (list, tuple)) or len(direction) != 2:
        raise ConfigError("flight.flow_direction must contain two components")
    try:
        dx, dy = map(float, direction)
    except (ValueError, TypeError) as exc:
        raise ConfigError("flight.flow_direction must be numeric") from exc
    mag = math.hypot(dx, dy)
    if not math.isfinite(mag) or mag == 0:
        raise ConfigError("flight.flow_direction must be finite and nonzero")
    cfg["flight"]["flow_direction"] = [dx / mag, dy / mag]
    if cfg["thermal"]["mode"] != "adiabatic":
        raise ConfigError("This study requires thermal.mode=adiabatic (zero wall heat flux). "
                          "Run study.py migrate --config <case.yaml> to migrate an old config.")
    if set(cfg["thermal"]) - {"mode"}:
        raise ConfigError("Adiabatic configuration must contain only thermal.mode; "
                          "remove prescribed wall temperatures and radiation settings using study.py migrate.")
    if cfg["geometry"].get("input_kind", "prepared_fluid_surface") != "prepared_fluid_surface":
        raise ConfigError("Expected one prepared XY fluid surface: farfield minus body hole")
    if cfg.get("mesh", {}).get("length_unit", "m") != "m":
        raise ConfigError("This workflow expects geometry and meshing lengths in metres")
    runtime = cfg.get("runtime", {})
    processors = positive("runtime.processors", runtime.get("processors", 4))
    if not processors.is_integer():
        raise ConfigError("runtime.processors must be an integer")
    iterations = positive("solver.max_iterations", cfg["solver"]["max_iterations"])
    if not iterations.is_integer():
        raise ConfigError("solver.max_iterations must be an integer")
    interval = positive("solver.temperature_history.sample_every_iterations",
                        cfg["solver"].get("temperature_history", {}).get("sample_every_iterations", 25))
    if not interval.is_integer():
        raise ConfigError("solver.temperature_history.sample_every_iterations must be an integer")
    positive("solver.courant_number",cfg["solver"].get("courant_number",.1))
    if cfg["solver"].get("initialization","freestream") not in ("freestream","hybrid"):
        raise ConfigError("solver.initialization must be freestream or hybrid")
    for name in ("geometry_start_timeout_s", "fluent_start_timeout_s"):
        positive("runtime." + name, runtime.get(name, 300))
    release_delay = float(runtime.get("meshing_release_delay_s", 10))
    if not math.isfinite(release_delay) or release_delay < 0 or release_delay > 60:
        raise ConfigError("runtime.meshing_release_delay_s must be in [0, 60]")
    if cfg.get("thermal_export", {}).get("enabled", False):
        raise ConfigError("Heat-load export is incompatible with this adiabatic study; migrate the config.")
    export = cfg.get("temperature_export", {})
    positive("temperature_export.topology_tolerance_m", export.get("topology_tolerance_m", 1e-9))
    if cfg.get("aero", {}).get("enabled", True):
        from .aero import reference_state
        reference_state(cfg)
    return cfg


def load_config(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"Config must be a YAML file: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config must contain a mapping: {path}")
    validate_config(cfg)
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    for section, key in (("project", "work_dir"), ("geometry", "input_file")):
        target = Path(cfg[section][key])
        if not target.is_absolute():
            target = path.parent / target
        cfg[section][key] = str(target.resolve())
    return cfg
