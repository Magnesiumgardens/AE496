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


def _validate_solver(solver, prefix="solver", native=False):
    iterations = positive(prefix + ".max_iterations", solver.get("max_iterations"))
    if not iterations.is_integer():
        raise ConfigError(prefix + ".max_iterations must be an integer")
    interval = positive(prefix + ".temperature_history.sample_every_iterations",
                        solver.get("temperature_history", {}).get("sample_every_iterations", 25))
    if not interval.is_integer():
        raise ConfigError(prefix + ".temperature_history.sample_every_iterations must be an integer")
    courant = solver.get("courant_number", .1)
    if courant is not None:
        positive(prefix + ".courant_number", courant)
    elif not native:
        raise ConfigError(prefix + ".courant_number=null requires a workflow with native startup controls")
    scheme = solver.get("flow_scheme", "first-order-upwind")
    if scheme is None and not native:
        raise ConfigError(prefix + ".flow_scheme=null requires a workflow with native startup controls")
    if scheme is not None and (not isinstance(scheme, str) or not scheme.strip()):
        raise ConfigError(prefix + ".flow_scheme must be a nonempty scheme name or null")
    thresholds = solver.get("residual_thresholds", {})
    if not isinstance(thresholds, dict):
        raise ConfigError(prefix + ".residual_thresholds must contain equation thresholds")
    for equation, threshold in thresholds.items():
        positive(prefix + ".residual_thresholds." + str(equation), threshold)
    if solver.get("initialization", "freestream") not in ("freestream", "hybrid"):
        raise ConfigError(prefix + ".initialization must be freestream or hybrid")


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
    workflow = cfg.get("workflow", {})
    if not isinstance(workflow, dict):
        raise ConfigError("workflow must contain a mapping")
    strategy = workflow.get("strategy")
    if strategy not in (None, "two_endpoint", "single_solve"):
        raise ConfigError("workflow.strategy must be single_solve or two_endpoint when specified")
    if strategy == "single_solve":
        calibration = cfg.get("_calibration_kind") == "cold_wall" and cfg.get("_study_work_dir")
        if calibration:
            if cfg["thermal"]["mode"] != "fixed_temperature":
                raise ConfigError("The marked cold calibration requires fixed_temperature")
            positive("thermal.wall_temperature_k", _required(cfg, "thermal.wall_temperature_k"))
        elif cfg["thermal"]["mode"] != "radiative_equilibrium":
            raise ConfigError("The single-solve workflow requires thermal.mode=radiative_equilibrium")
        emissivity = positive("thermal.emissivity", cfg["thermal"].get("emissivity", .9) if calibration else _required(cfg, "thermal.emissivity"))
        if emissivity > 1:
            raise ConfigError("thermal.emissivity must be greater than zero and at most one")
        positive("thermal.external_radiation_temperature_k",
                 cfg["thermal"].get("external_radiation_temperature_k", 3) if calibration else _required(cfg, "thermal.external_radiation_temperature_k"))
        if "wall_temperature_k" in cfg["thermal"]:
            positive("thermal.wall_temperature_k (reference only)", cfg["thermal"]["wall_temperature_k"])
    elif strategy == "two_endpoint":
        internal_adiabatic = cfg.get("_endpoint_kind") == "adiabatic" and cfg.get("_study_work_dir")
        if internal_adiabatic:
            if cfg["thermal"] != {"mode": "adiabatic"}:
                raise ConfigError("The marked adiabatic endpoint must contain only thermal.mode=adiabatic")
        elif cfg["thermal"]["mode"] != "fixed_temperature":
            raise ConfigError("The two-endpoint workflow starts with thermal.mode=fixed_temperature; "
                              "the warm-started adiabatic reference is a separate stage.")
        if not internal_adiabatic:
            positive("thermal.wall_temperature_k", _required(cfg, "thermal.wall_temperature_k"))
        reference_iterations = positive("workflow.adiabatic_reference_iterations",
                                        cfg["workflow"].get("adiabatic_reference_iterations", 600))
        if not reference_iterations.is_integer():
            raise ConfigError("workflow.adiabatic_reference_iterations must be an integer")
        positive("workflow.min_delta_temperature_k", workflow.get("min_delta_temperature_k", 25))
        positive("workflow.coordinate_tolerance_m", workflow.get("coordinate_tolerance_m", 1e-9))
        for endpoint in ("cold_wall", "adiabatic"):
            name = endpoint + "_solver"
            overrides = workflow.get(name, {})
            if not isinstance(overrides, dict):
                raise ConfigError("workflow." + name + " must contain solver overrides")
            _validate_solver(dict(cfg["solver"], **overrides), "workflow." + name, native=True)
    else:
        # Keep earlier direct-adiabatic cases readable without silently changing
        # their physical model. The updater performs the explicit migration.
        if cfg["thermal"]["mode"] != "adiabatic":
            raise ConfigError("Without workflow.strategy, this case requires thermal.mode=adiabatic. "
                              "Run study.py migrate --config <case.yaml> to select the single-solve workflow.")
        if set(cfg["thermal"]) - {"mode"}:
            raise ConfigError("A direct adiabatic configuration must contain only thermal.mode")
    if cfg["geometry"].get("input_kind", "prepared_fluid_surface") != "prepared_fluid_surface":
        raise ConfigError("Expected one prepared XY fluid surface: farfield minus body hole")
    if cfg.get("mesh", {}).get("length_unit", "m") != "m":
        raise ConfigError("This workflow expects geometry and meshing lengths in metres")
    runtime = cfg.get("runtime", {})
    processors = positive("runtime.processors", runtime.get("processors", 4))
    if not processors.is_integer():
        raise ConfigError("runtime.processors must be an integer")
    _validate_solver(cfg["solver"], native=strategy in ("two_endpoint", "single_solve"))
    for name in ("geometry_start_timeout_s", "fluent_start_timeout_s"):
        positive("runtime." + name, runtime.get(name, 300))
    release_delay = float(runtime.get("meshing_release_delay_s", 10))
    if not math.isfinite(release_delay) or release_delay < 0 or release_delay > 60:
        raise ConfigError("runtime.meshing_release_delay_s must be in [0, 60]")
    if strategy not in ("two_endpoint", "single_solve") and cfg.get("thermal_export", {}).get("enabled", False):
        raise ConfigError("Heat-load export is incompatible with this adiabatic study; migrate the config.")
    if strategy == "two_endpoint":
        cold_export = cfg.get("thermal_export", {})
        sign = cold_export.get("heat_flux_into_wall_sign", -1.0)
        if sign not in (-1, 1):
            raise ConfigError("thermal_export.heat_flux_into_wall_sign must be -1 or +1")
        positive("thermal_export.topology_tolerance_m", cold_export.get("topology_tolerance_m", 1e-9))
        positive("thermal_export.min_delta_temperature_k", cold_export.get("min_delta_temperature_k", 25.0))
    export = cfg.get("temperature_export", {})
    positive("temperature_export.topology_tolerance_m", export.get("topology_tolerance_m", 1e-9))
    if cfg.get("aero", {}).get("enabled", True):
        from .aero import reference_state
        reference_state(cfg)
        from .force_history import settings as force_history_settings
        try:force_history_settings(cfg)
        except ValueError as exc:raise ConfigError(str(exc)) from exc
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
    study_root = Path(cfg.get("_study_work_dir", cfg["project"]["work_dir"]))
    if not study_root.is_absolute():
        study_root = path.parent / study_root
    study_root = study_root.resolve()
    if cfg.get("_study_work_dir"):
        cfg["_study_work_dir"] = str(study_root)
    directory = cfg.get("temperature_export", {}).get("output_directory", "temperature")
    if not isinstance(directory, str) or not directory.strip():
        raise ConfigError("temperature_export.output_directory must be a directory inside the selected case")
    output = Path(directory)
    if not output.is_absolute():
        output = study_root / output
    output = output.resolve()
    if output == study_root or study_root not in output.parents:
        raise ConfigError("temperature_export.output_directory must be a subdirectory inside the selected case")
    return cfg
