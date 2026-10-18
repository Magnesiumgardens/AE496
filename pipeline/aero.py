"""Planar sectional force reports for Fluent 2025 R1.

Reference depth is explicitly one metre. No whole-vehicle area is inferred.
"""
from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
import re

MODEL_VERSION = "2.0-planar-section-components"


def positive(name, value):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def reference_state(cfg):
    aero = cfg.get("aero", {})
    if aero.get("model_kind", "planar_section") != "planar_section":
        raise ValueError("This force exporter supports planar_section only, not axisymmetric noses")
    flight = cfg["flight"]
    direction = flight["flow_direction"]
    if len(direction) != 2:
        raise ValueError("flight.flow_direction must have two components")
    dx, dy = map(float, direction)
    magnitude = math.hypot(dx, dy)
    if not math.isfinite(magnitude) or magnitude == 0:
        raise ValueError("flight.flow_direction must be finite and nonzero")
    dx, dy = dx / magnitude, dy / magnitude
    p = positive("flight.static_pressure_pa", flight["static_pressure_pa"])
    t = positive("flight.static_temperature_k", flight["static_temperature_k"])
    mach = positive("flight.mach", flight["mach"])
    gamma = positive("aero.ratio_of_specific_heats", aero.get("ratio_of_specific_heats", 1.4))
    if gamma <= 1:
        raise ValueError("aero.ratio_of_specific_heats must exceed one")
    gas_r = positive("aero.gas_constant_j_kgk", aero.get("gas_constant_j_kgk", 287.05))
    width = aero.get("reference_area_per_unit_span_m")
    length = aero.get("reference_length_m")
    if width is not None:
        width = positive("aero.reference_area_per_unit_span_m", width)
    if length is not None:
        length = positive("aero.reference_length_m", length)
    return {
        "model_kind": "planar_section", "mach": mach,
        "static_pressure_Pa": p, "static_temperature_K": t,
        "density_kg_m3": p / (gas_r * t),
        "velocity_m_s": mach * math.sqrt(gamma * gas_r * t),
        "dynamic_pressure_Pa": 0.5 * gamma * p * mach ** 2,
        "ratio_of_specific_heats": gamma, "gas_constant_J_kgK": gas_r,
        "drag_direction": [dx, dy, 0.0], "lift_direction": [-dy, dx, 0.0],
        "report_depth_m": 1.0,
        "reference_area_per_unit_span_m": width,
        "reference_length_m": length,
        "physics_signature": cfg.get("physics", {}),
        "wall_thermal_signature": cfg.get("thermal", {}),
    }


def report_scalar(raw, target):
    """Read one named report without mistaking iteration indices for forces.

    Fluent 251 returns named [value, integer metadata] records; clients can
    also separate values and metadata. Never search metadata for a force.
    """
    values = raw[0] if isinstance(raw, (tuple, list)) else raw
    if not isinstance(values, dict) or target not in values:
        raise ValueError(f"No value returned for force report {target!r}")
    value = values[target]
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            value = value[0]
        elif (len(value) == 2 and isinstance(value[1], Real)
              and not isinstance(value[1], bool) and math.isfinite(value[1])
              and float(value[1]).is_integer() and value[1] >= 0):
            value = value[0]
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"Invalid or ambiguous force value for {target!r}: {value!r}")
    return float(value)


def report_iteration(raw, target):
    """Legacy extraction of integer report metadata, or ``None``.

    This layout does not establish that the integer counts completed solver
    iterations. Force-history verification uses the native file's labeled
    iteration column instead; a force value is never used as a counter.
    """
    values = raw[0] if isinstance(raw, (tuple, list)) and raw else raw
    if not isinstance(values, dict) or target not in values:
        return None
    record = values[target]
    candidate = None
    if isinstance(record, (tuple, list)) and len(record) == 2:
        candidate = record[1]
    elif isinstance(raw, (tuple, list)) and len(raw) == 2 and isinstance(raw[1], dict):
        candidate = raw[1].get(target)
        if isinstance(candidate, (tuple, list)) and len(candidate) == 1:
            candidate = candidate[0]
    if candidate is None:
        return None
    if (isinstance(candidate, bool) or not isinstance(candidate, Real)
            or not math.isfinite(candidate) or not float(candidate).is_integer()
            or candidate < 0):
        raise ValueError(f"Invalid iteration metadata for force report {target!r}: {candidate!r}")
    return int(candidate)


def _force_report(solver, kind, zones, direction):
    definitions = solver.settings.solution.report_definitions
    collection = getattr(definitions, kind)
    name = "nose-study-" + kind
    if name not in collection.get_object_names():
        collection.create(name=name)
    report = collection[name]
    # Typed drag/lift collections determine report type. That child is inactive
    # in 251. Reuse saved reports; no destructive delete/recreate is necessary.
    report.zones = zones
    report.force_vector = direction
    report.per_zone = False
    # Never accidentally interpret a default coefficient as a dimensional force.
    allowed = report.report_output_type.allowed_values()
    force_type = next((v for v in allowed if str(v).casefold().replace(" ", "")
                       in ("force", kind + "force")), None)
    if force_type is None:
        raise ValueError(f"Fluent {kind} report does not expose Force output: {allowed}")
    report.report_output_type = force_type
    averaging = getattr(report, "average_over", None)
    if averaging is not None and (not hasattr(averaging, "is_active") or averaging.is_active()):
        report.average_over = 1
    return name


def setting_scalar(value):
    """Read constant numerical boundary groups from the pinned 251 API."""
    while isinstance(value, dict):
        option = str(value.get("option", "value")).casefold()
        if option not in ("value", "constant") or "value" not in value:
            raise ValueError(f"Expected a constant numerical boundary setting: {value!r}")
        value = value["value"]
    if isinstance(value, bool):
        raise ValueError("A boundary setting cannot be a boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Boundary setting is not finite")
    return number


def wall_thermal_state(thermal, requested):
    """Validate the actual boundary, including external-only wall radiation.

    Fluent's volumetric radiation model stays disabled: the wall Radiation
    condition independently balances gas-side heating against external emission.
    """
    mode = requested.get("mode")
    expected = {"adiabatic": "heatflux", "fixed_temperature": "temperature",
                "radiative_equilibrium": "radiation"}.get(mode)
    if expected is None:
        raise ValueError(f"Unsupported saved wall thermal mode: {mode!r}")
    condition = thermal.thermal_condition.get_state()
    if str(condition).casefold().replace(" ", "") != expected:
        raise ValueError(f"Saved wall condition differs from {mode}; rerun CFD")

    def check(name, setting, value):
        actual = setting_scalar(setting.get_state())
        wanted = setting_scalar(value)
        if not math.isclose(actual, wanted, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"Saved case {name}={actual} differs from requested {wanted}; rerun CFD")
        return actual

    state = {"thermal_condition": str(condition)}
    if mode == "adiabatic":
        state["heat_flux_W_m2"] = check("wall heat flux", thermal.heat_flux, 0.0)
    elif mode == "fixed_temperature":
        state["wall_temperature_k"] = check("cold wall temperature", thermal.temperature,
                                             requested["wall_temperature_k"])
    else:
        state["emissivity"] = check("external emissivity", thermal.external_emissivity,
                                     requested["emissivity"])
        state["external_radiation_temperature_k"] = check(
            "external radiation temperature", thermal.ext_rad_temperature,
            requested["external_radiation_temperature_k"])
    return state


def validate_saved_conditions(solver, cfg):
    """Check saved physical conditions before relabeling an existing CFD result."""
    setup = solver.settings.setup
    mode = cfg["thermal"].get("mode")
    if mode not in ("adiabatic", "fixed_temperature", "radiative_equilibrium"):
        raise ValueError("Saved-result export requires a supported wall thermal configuration")
    if not setup.models.energy.enabled.get_state():
        raise ValueError("Saved case has no fluid energy equation; rerun CFD")
    if str(setup.models.radiation.model.get_state()).casefold() != "none":
        raise ValueError("Saved case has volumetric radiation enabled; rerun CFD with radiation.model=none")
    state = reference_state(cfg)
    base = cfg["geometry"].get("farfield_name", "farfield")
    farfields = setup.boundary_conditions.pressure_far_field
    zones = [z for z in farfields.get_object_names() if z == base or z.startswith(base + ".")]
    if not zones:
        raise ValueError("Saved case has no matching pressure-far-field boundaries")

    def check(name, actual, expected):
        if not math.isclose(setting_scalar(actual), setting_scalar(expected), rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"Saved case {name}={actual} differs from requested {expected}; rerun CFD")

    operating = setup.general.operating_conditions.operating_pressure.get_state()
    check("operating pressure", operating, cfg["physics"].get("operating_pressure_pa",0.0))
    for zone in zones:
        far = farfields[zone]
        check("Mach", far.momentum.mach_number.get_state(), state["mach"])
        check("static pressure", setting_scalar(far.momentum.gauge_pressure.get_state()) + setting_scalar(operating),
              state["static_pressure_Pa"])
        check("temperature", far.thermal.temperature.get_state(), state["static_temperature_K"])
        flow = far.momentum.flow_direction.get_state()
        mag = math.hypot(setting_scalar(flow[0]), setting_scalar(flow[1]))
        if not math.isfinite(mag) or mag == 0:
            raise ValueError("Saved case freestream direction is invalid")
        for i in range(2):
            check("flow direction", setting_scalar(flow[i]) / mag, state["drag_direction"][i])
    model = setup.models.viscous.model.get_state()
    if model != cfg["physics"].get("viscous_model", "laminar"):
        raise ValueError("Saved case viscous model differs from config; rerun CFD")
    air_cfg = cfg["physics"]["air"]
    air = setup.materials.fluid["air"]
    for key, default in (("density", "ideal-gas"), ("viscosity", "sutherland")):
        if getattr(air, key).option.get_state() != air_cfg.get(key + "_model", default):
            raise ValueError(f"Saved case air {key} model differs from config; rerun CFD")
    if air_cfg.get("viscosity_model", "sutherland") == "sutherland":
        for field, config_key in (
            ("reference_viscosity", "sutherland_reference_viscosity"),
            ("reference_temperature", "sutherland_reference_temperature_k"),
            ("effective_temperature", "sutherland_effective_temperature_k"),
        ):
            if config_key in air_cfg:
                check("Sutherland " + field, getattr(air.viscosity.sutherland, field).get_state(),
                      air_cfg[config_key])
    walls = setup.boundary_conditions.wall
    base = cfg["geometry"].get("body_wall_name", "body_wall")
    zones = [z for z in walls.get_object_names() if z == base or z.startswith(base + ".")]
    if not zones:
        raise ValueError("Saved case has no matching body-wall boundaries")
    for name in zones:
        wall_thermal_state(walls[name].thermal, cfg["thermal"])


def configure_force_reports(solver, cfg, wall_zones):
    """Set one-metre-depth whole-body dimensional drag/lift definitions.

    Shared by final export and the native per-iteration report-file monitor.
    The caller supplies the complete matched body-wall zone collection.
    """
    state = reference_state(cfg)
    base = cfg["geometry"].get("body_wall_name", "body_wall")
    zones = [z for z in wall_zones if z == base or z.startswith(base + ".")]
    if not zones or len(zones) != len(wall_zones):
        raise ValueError("Force integration requires only matching body-wall zones")
    if str(solver.settings.setup.general.solver.two_dim_space.get_state()).casefold() != "planar":
        raise ValueError("Force-per-span normalization requires Fluent planar mode")
    refs = solver.settings.setup.reference_values
    refs.depth = 1.0
    refs.density = state["density_kg_m3"]
    refs.velocity = state["velocity_m_s"]
    refs.pressure = state["static_pressure_Pa"]
    refs.temperature = state["static_temperature_K"]
    if state["reference_area_per_unit_span_m"] is not None:
        refs.area = state["reference_area_per_unit_span_m"] * state["report_depth_m"]
    if state["reference_length_m"] is not None:
        refs.length = state["reference_length_m"]
    state["body_wall_zones"] = zones
    names = {kind: _force_report(solver, kind, zones, state[kind + "_direction"])
             for kind in ("drag", "lift")}
    return names, state


def parse_force_components(text, zones):
    """Read dimensional pressure/viscous/total columns from a Fluent table.

    Fluent's native force report also prints coefficient columns. Only the
    first three numeric columns under an explicit ``Forces [N]`` header are
    accepted; coefficients, per-zone rows and Net are never double-counted.
    This deliberately rejects unknown layouts rather than inventing a split.
    """
    if not re.search(r"\bForces?\s*[\[(]\s*N\s*[\])]", text, re.IGNORECASE):
        raise ValueError("Native force report has no dimensional Forces [N] header")
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if re.search(
        r"\bPressure\s+Viscous\s+Total\b", line, re.IGNORECASE)]
    if len(headers) != 1:
        raise ValueError("Native force report has no unambiguous Pressure Viscous Total table")
    headings = re.findall(r"\b(?:Pressure|Viscous|Total)\b", lines[headers[0]], re.IGNORECASE)
    if [value.casefold() for value in headings[:3]] != ["pressure", "viscous", "total"]:
        raise ValueError("Native force report uses an unsupported force-column order")
    rows = {}
    net = None
    for line in lines[headers[0] + 1:]:
        fields = line.strip().split()
        if not fields:
            continue
        label = fields[0].strip('"')
        if label not in zones and label.casefold() != "net":
            continue
        try:
            values = [float(value.replace("D", "E").replace("d", "e"))
                      for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Non-numeric force row for {label!r}") from exc
        if len(values) not in (3, 6) or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Unsupported force row for {label!r}: {fields[1:]!r}")
        if label.casefold() == "net":
            if net is not None:
                raise ValueError("Native force report contains multiple Net rows")
            net = values[:3]
        else:
            if label in rows:
                raise ValueError(f"Native force report repeats body zone {label!r}")
            rows[label] = values[:3]
    if set(rows) != set(zones):
        raise ValueError("Native force report does not list every requested body-wall zone")
    summed = [sum(row[i] for row in rows.values()) for i in range(3)]
    tolerance = max(1e-6, 1e-5 * max(abs(value) for value in (*summed, *(net or []))))
    if net is not None and any(abs(a - b) > tolerance for a, b in zip(net, summed)):
        raise ValueError("Native force Net row differs from the sum over body-wall zones")
    pressure, viscous, total = net if net is not None else summed
    if abs(pressure + viscous - total) > tolerance:
        raise ValueError("Native pressure and viscous forces do not sum to native total")
    return pressure, viscous, total


def _drag_components(solver, cfg, state, total_drag):
    """Use the supported 251 native report, not nonexistent force-type knobs."""
    result = {
        "pressure_drag_per_unit_span_N_m": None,
        "viscous_drag_per_unit_span_N_m": None,
        "component_status": "unavailable",
    }
    try:
        work = cfg.get("project", {}).get("work_dir")
        if not work:
            raise ValueError("A project work directory is required for the native force table")
        path = Path(work).resolve() / "aero_forces_drag.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        native = solver.settings.results.report.forces
        option = "Forces"
        argument = getattr(native, "option", None)
        if argument is not None and hasattr(argument, "allowed_values"):
            allowed = argument.allowed_values()
            option = next((v for v in allowed if str(v).casefold() in ("forces", "force")), None)
            if option is None:
                raise ValueError(f"Native report does not expose Forces: {allowed!r}")
        # Avoid consuming stale output if the native command fails before write.
        path.unlink(missing_ok=True)
        native(option=option, wall_zones=state["body_wall_zones"],
               direction_vector=state["drag_direction"], write_to_file=True,
               file_name=str(path), append_data=False)
        pressure, viscous, native_total = parse_force_components(
            path.read_text(encoding="utf-8-sig"), state["body_wall_zones"])
        depth = state["report_depth_m"]
        pressure, viscous, native_total = (v / depth for v in (pressure, viscous, native_total))
        tolerance = max(1e-6, 1e-5 * max(abs(v) for v in
                                       (pressure, viscous, native_total, total_drag)))
        if abs(native_total - total_drag) > tolerance:
            raise ValueError("Native force table total differs from the dimensional drag report")
        result.update(
            pressure_drag_per_unit_span_N_m=pressure,
            viscous_drag_per_unit_span_N_m=viscous,
            component_status="complete",
            component_method="Fluent results.report.forces dimensional table",
            component_report_file=str(path),
            drag_component_balance={
                "status": "passed", "pressure_plus_viscous_N_m": pressure + viscous,
                "native_total_N_m": native_total, "total_drag_N_m": total_drag,
                "difference_N_m": pressure + viscous - total_drag,
                "tolerance_N_m": tolerance,
                "note": "Tolerance allows rounded native table output; signed forces are retained.",
            },
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        result["component_error"] = str(exc)
    return result


def aerodynamic_summary(solver, cfg, wall_zones):
    from .force_history import assess_saved_force_history

    names, state = configure_force_reports(solver, cfg, wall_zones)
    # Query separately: 251 can return separate mappings for a multi-report
    # request, which is ambiguous with clients' value/iteration mappings.
    compute = solver.settings.solution.report_definitions.compute
    drag, lift = [report_scalar(compute(report_defs=[names[kind]]), names[kind])
                  / state["report_depth_m"] for kind in ("drag", "lift")]
    components = _drag_components(solver, cfg, state, drag)
    width = state["reference_area_per_unit_span_m"]
    scale = state["dynamic_pressure_Pa"] * width if width is not None else None
    return {
        "status": "complete", "model_version": MODEL_VERSION,
        **state, **components,
        "drag_per_unit_span_N_m": drag, "lift_per_unit_span_N_m": lift,
        "total_drag_per_unit_span_N_m": drag,
        "Cd": drag / scale if scale is not None else None,
        "Cd_pressure": (components["pressure_drag_per_unit_span_N_m"] / scale
                        if scale is not None and components["component_status"] == "complete" else None),
        "Cd_viscous": (components["viscous_drag_per_unit_span_N_m"] / scale
                       if scale is not None and components["component_status"] == "complete" else None),
        "Cl": lift / scale if scale is not None else None,
        "L_over_D": lift / drag if drag != 0 else None,
        "coefficient_status": "computed" if scale is not None else "reference_width_required",
        "force_stability": assess_saved_force_history(cfg),
        "convergence_status": "not_verified",
        "limitations": [
            "Sectional planar forces are not whole-vehicle or axisymmetric forces.",
            "Total drag includes pressure and viscous forces; the split requires a validated native force table.",
            "Freestream normalization assumes calorically perfect air with configured gamma and R.",
            "Iterations completed is not evidence of convergence or mesh independence.",
        ],
    }


def require_force_stability(cfg, result):
    """Block acceptance when explicitly requested, while retaining measured forces."""
    required = cfg.get("aero", {}).get("force_history", {}).get("require_stable", False)
    if (cfg.get("aero", {}).get("enabled", True) and required
            and result.get("force_stability", {}).get("status") != "passed"):
        status = result.get("force_stability", {}).get("status", "not_verified")
        reason = result.get("force_stability", {}).get("reason", "No failure detail was recorded")
        case_arg = '"' + str(cfg.get("_config_path", "<case.yaml>")) + '"'
        advice = (f"Use study.py continue --config {case_arg} --iterations 200 --stop-after aero "
                  "to record additional iterations without reinitializing the flow."
                  if status in ("failed", "insufficient_history") else
                  f"Run study.py force-status --config {case_arg} to inspect the recording and verification details.")
        raise ValueError(
            f"Drag force-stability screen is {status}: {reason}. "
            f"Measured forces and the saved CFD pair are retained. {advice}")
    if (cfg.get("aero", {}).get("enabled", True) and required
            and result.get("residual_status") != "residual_screen_passed"):
        raise ValueError("The aerodynamic result also requires a matching passed residual screen; measured forces are retained")
