"""Planar sectional force reports for Fluent 2025 R1.

Reference depth is explicitly one metre. No whole-vehicle area is inferred.
"""
from __future__ import annotations

import math
from numbers import Real

MODEL_VERSION = "1.0-planar-section"


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

    Fluent 251 returns named [value, iteration] records; newer clients can
    separate values and iteration metadata. Never search metadata for a value.
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


def aerodynamic_summary(solver, cfg, wall_zones):
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
    names = [_force_report(solver, kind, zones, state[kind + "_direction"])
             for kind in ("drag", "lift")]
    # Query separately: 251 can return separate mappings for a multi-report
    # request, which is ambiguous with clients' value/iteration mappings.
    compute = solver.settings.solution.report_definitions.compute
    drag, lift = [report_scalar(compute(report_defs=[name]), name)
                  / state["report_depth_m"] for name in names]
    width = state["reference_area_per_unit_span_m"]
    scale = state["dynamic_pressure_Pa"] * width if width is not None else None
    return {
        "status": "complete", "model_version": MODEL_VERSION,
        **state, "body_wall_zones": zones,
        "drag_per_unit_span_N_m": drag, "lift_per_unit_span_N_m": lift,
        "Cd": drag / scale if scale is not None else None,
        "Cl": lift / scale if scale is not None else None,
        "L_over_D": lift / drag if drag != 0 else None,
        "coefficient_status": "computed" if scale is not None else "reference_width_required",
        "convergence_status": "not_verified",
        "limitations": [
            "Sectional planar forces are not whole-vehicle or axisymmetric forces.",
            "Total drag includes pressure and viscous contributions; no separate split is reported.",
            "Freestream normalization assumes calorically perfect air with configured gamma and R.",
            "Iterations completed is not evidence of convergence or mesh independence.",
        ],
    }
