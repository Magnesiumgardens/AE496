"""Thermal screening metrics and comparisons; no weighted winner is invented."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .aero import MODEL_VERSION, positive, reference_state

THERMAL_MODEL_VERSION = "1.1-fixed-heat-flux-screening"
THERMAL_METRIC_FIELDS = (
    "peak_temperature_K", "time_above_temperature_limit_s",
    "temperature_limit_margin_K", "temperature_limit_status",
    "cold_wall_heating_power_per_unit_span_W_m",
    "applied_heat_load_per_unit_span_J_m",
    "solid_area_m2", "solid_mass_per_unit_span_kg_m",
)
AERO_FIELDS = (
    "drag_per_unit_span_N_m", "total_drag_per_unit_span_N_m",
    "pressure_drag_per_unit_span_N_m", "viscous_drag_per_unit_span_N_m",
    "lift_per_unit_span_N_m", "Cd", "Cd_pressure", "Cd_viscous", "Cl", "L_over_D",
    "dynamic_pressure_Pa", "reference_area_per_unit_span_m", "reference_length_m",
    "component_status", "force_stability_status", "residual_status",
)


def aero_report_fields(aero):
    """Flatten persisted force metrics without fabricating missing components."""
    fields = {key: aero.get(key) for key in AERO_FIELDS}
    stability = aero.get("force_stability", {})
    fields["force_stability_status"] = (stability.get("status", "not_verified")
        if isinstance(stability, dict) else "not_verified")
    fields["component_status"] = aero.get("component_status", "unavailable")
    fields["residual_status"] = aero.get("residual_status", "not_verified")
    return fields


def force_screening_eligible(row):
    """A stable force history is required for a Pareto screening claim.

    Passing this check alone does not establish residual convergence or mesh
    independence. Unverified dimensional forces remain in exported tables.
    """
    return (row.get("force_stability_status") == "passed"
            and (row.get("residual_status") == "residual_screen_passed"
                 or row.get("convergence_status") == "residual_and_force_screens_passed"))


def thermal_screening_metrics(xy, flux, history, settings, temperature_limit=None):
    """Integrate the reconstructed polygon, not a claimed exact CAD surface.

    Heat load is the prescribed fixed incoming load before radiation losses.
    Time above a limit uses piecewise-linear saved maximum-temperature history.
    """
    if len(xy) < 3 or len(xy) != len(flux):
        raise ValueError("Need a closed polygon with one flux value per point")
    power = area2 = 0.0
    for i, (x, y) in enumerate(xy):
        nx, ny = xy[(i + 1) % len(xy)]
        q0, q1 = float(flux[i]), float(flux[(i + 1) % len(flux)])
        if not all(math.isfinite(float(v)) for v in (x, y, nx, ny, q0, q1)):
            raise ValueError("Polygon and flux must be finite")
        power += 0.5 * (q0 + q1) * math.hypot(nx - x, ny - y)
        area2 += x * ny - nx * y
    area = abs(area2) / 2
    duration = positive("duration", settings["duration"])
    density = positive("density", settings["rho"])
    initial = positive("initial temperature", settings["initial_t"])
    points = [(0.0, initial)]
    for row in history:
        time, maximum = float(row["time_s"]), float(row["maximum_K"])
        if not math.isfinite(time) or not math.isfinite(maximum) or time <= points[-1][0]:
            raise ValueError("Thermal history must have finite, strictly increasing positive times")
        points.append((time, maximum))
    if len(points) < 2 or not math.isclose(points[-1][0], duration, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Thermal history does not cover the requested duration")
    peak = max(v for _, v in points)
    result = {
        "peak_temperature_K": peak,
        "time_above_temperature_limit_s": None,
        "temperature_limit_margin_K": None,
        "temperature_limit_status": "not_configured",
        "cold_wall_heating_power_per_unit_span_W_m": power,
        "applied_heat_load_per_unit_span_J_m": power * duration,
        "solid_area_m2": area,
        "solid_mass_per_unit_span_kg_m": area * density,
        "notes": [
            "Heat load is signed prescribed fixed-q input before radiation; not endpoint convection.",
            "Mass assumes a filled uniform-material planar section reconstructed from wall centroids.",
            "Time above limit interpolates saved spatial-maximum temperatures, not individual node exposure.",
        ],
    }
    if temperature_limit is not None:
        limit = positive("design_limits.max_temperature_k", temperature_limit)
        above = 0.0
        for (t0, v0), (t1, v1) in zip(points, points[1:]):
            dt = t1 - t0
            if v0 > limit and v1 > limit:
                above += dt
            elif (v0 > limit) != (v1 > limit):
                fraction = (limit - v0) / (v1 - v0)
                above += dt * (fraction if v0 > limit else 1 - fraction)
        result.update(
            time_above_temperature_limit_s=above,
            temperature_limit_margin_K=limit - peak,
            temperature_limit_status="within_limit" if peak <= limit else "exceeded",
        )
    return result


def read_aero(case_dir):
    try:
        return json.loads((Path(case_dir) / "aero_summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def aero_complete(case_dir, cfg):
    """Reject cached forces after changed case/data, reference values or direction."""
    directory = Path(case_dir)
    result = read_aero(directory)
    if result.get("status") != "complete" or result.get("model_version") != MODEL_VERSION:
        return False
    expected = reference_state(cfg)
    if any(result.get(key) != value for key, value in expected.items()):
        return False
    base = cfg["geometry"].get("body_wall_name", "body_wall")
    zones = result.get("body_wall_zones", [])
    if not zones or any(not (z == base or z.startswith(base + ".")) for z in zones):
        return False
    # Availability is a reported scientific status, not cache freshness: an
    # unchanged unavailable history may still accompany useful dimensional
    # forces. Changed history/settings must, however, refresh that status.
    from .force_history import assess_saved_force_history
    if result.get("force_stability") != assess_saved_force_history(cfg):
        return False
    try:
        saved = (directory / "aero_summary.json").stat().st_mtime_ns
        return all((directory / name).stat().st_mtime_ns <= saved
                   for name in ("04_final.cas.h5", "04_final.dat.h5"))
    except OSError:
        return False


def comparison_key(aero, boundary_layer):
    return tuple(json.dumps(aero.get(k), sort_keys=True) for k in (
        "model_kind", "mach", "static_pressure_Pa", "static_temperature_K",
        "ratio_of_specific_heats", "gas_constant_J_kgK", "drag_direction",
        "reference_area_per_unit_span_m", "reference_length_m",
        "physics_signature", "wall_thermal_signature",
    )) + (boundary_layer,)


def pareto_cases(rows):
    """Minimize drag and peak temperature, retaining identical tied designs."""
    candidates = []
    for row in rows:
        try:
            drag, temp = float(row["drag_per_unit_span_N_m"]), float(row["peak_temperature_K"])
        except (ValueError, TypeError, KeyError):
            continue
        if (row.get("status") == "complete" and row.get("aero_status") == "complete"
                and force_screening_eligible(row)
                and math.isfinite(drag) and drag >= 0 and math.isfinite(temp)
                and row.get("temperature_limit_status") != "exceeded"):
            candidates.append((row["case"], drag, temp))
    return [name for name, drag, temp in candidates if not any(
        d <= drag and t <= temp and (d < drag or t < temp)
        for _, d, t in candidates
    )]


def create_trade_outputs(runs_dir, rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = [r for r in rows if r.get("aero_status") == "complete" and r.get("status") == "complete"]
    keys = {(comparison_key(read_aero(Path(runs_dir) / r["case"]), r.get("boundary_layer_used")),
             r.get("thermal_loading_mode"), r.get("thermal_comparison_signature"))
            for r in eligible}
    comparable = len(keys) <= 1
    frontier = pareto_cases(eligible) if comparable else []
    unchecked = [dict(case=row["case"], force_stability_status=row.get(
        "force_stability_status", "not_verified"), residual_status=row.get("residual_status", "not_verified"))
        for row in eligible if not force_screening_eligible(row)]
    report = {
        "status": "screening_only" if comparable else "incomparable_conditions",
        "pareto_cases": frontier,
        "objectives": ["minimize drag per unit span", "minimize peak temperature"],
        "limits": "Only configured temperature limits are screened; no material limit is assumed.",
        "force_screening_requirement": "Drag stability and residual screens must both pass; this is not proof of full CFD convergence or mesh independence.",
        "cases_excluded_from_pareto_for_force_checks": unchecked,
        "limitations": [
            "No automatic best-nose selection; convergence and mesh independence remain unverified.",
            "Moments, stability, structural stress, material property variation and 3-D effects are not evaluated.",
        ],
    }
    path = output_dir / "nose_trade_screening.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    created = {"nose_trade_screening": str(path)}
    if not comparable:
        return created
    for key, label in (("drag_per_unit_span_N_m", "Sectional drag [N/m]"), ("Cd", "Drag coefficient")):
        valid = []
        for row in eligible:
            try:
                value = float(row[key])
            except (TypeError, ValueError, KeyError):
                continue
            if math.isfinite(value):
                valid.append((row, value))
        if not valid:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(valid)), 5))
        ax.bar([r["case"] for r, _ in valid], [v for _, v in valid])
        ax.set_ylabel(label)
        ax.set_title("Dimensional nose comparison — force checks shown below")
        ax.set_xticks(range(len(valid)), [r["case"] + "\n[" + str(r.get(
            "force_stability_status", "not_verified")) + "]" for r, _ in valid])
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        p = output_dir / f"{key}_by_geometry.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        created[key + "_by_geometry"] = str(p)
        for ykey, ylabel in (("peak_temperature_K", "Peak transient temperature [K]"),
                              ("max_heat_flux_W_m2", "Maximum cold-wall heat flux [W/m²]")):
            fig, ax = plt.subplots(figsize=(8, 5))
            count = 0
            for row, value in valid:
                try:
                    y = float(row[ykey])
                except (TypeError, ValueError, KeyError):
                    continue
                if not math.isfinite(y):
                    continue
                marker = "*" if row["case"] in frontier and ykey == "peak_temperature_K" else (
                    "o" if force_screening_eligible(row) else "x")
                ax.scatter(value, y, marker=marker, s=85)
                ax.annotate(row["case"], (value, y), xytext=(4, 4), textcoords="offset points")
                count += 1
            if count:
                ax.set(xlabel=label, ylabel=ylabel, title="Screening trade (* = Pareto candidate; × = force checks unverified/failed)")
                fig.tight_layout()
                p = output_dir / f"{ykey}_vs_{key}.png"
                fig.savefig(p, dpi=160)
                created[ykey + "_vs_" + key] = str(p)
            plt.close(fig)
    return created
