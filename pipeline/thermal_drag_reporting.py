"""Pair saved solid-heating metrics with current sectional drag; no solver run."""
from pathlib import Path

from .reporting import _plotting, _save, load_yaml, numeric, read_csv, read_json, write_csv
from .trade_study import aero_report_fields, force_screening_eligible, read_aero


def _thermal_primary_source(thermal):
    """Compare drag with the RE pair, retaining the cold pair for calibration.

    The actual calibrated thermal runner records re_case/re_data AND
    cold_case/cold_data. Drag records final_case/final_data from the RE run.
    Those whole dictionaries must not be compared for equality.
    """
    sources = thermal.get("source_sha256", {})
    if not isinstance(sources, dict):
        return {}
    if "re_case" in sources or "re_data" in sources:
        return {"final_case": sources.get("re_case"), "final_data": sources.get("re_data")}
    return {"final_case": sources.get("final_case"), "final_data": sources.get("final_data")}


def thermal_drag_outputs(runs_dir, cases, directory, colors):
    """Use the same net metrics as the existing heat-load and heat-flux bars."""
    from .gas_reporting import aero_complete_for_report
    from .mechanical_reporting import mechanical_complete

    directory = Path(directory)
    specifications = (
        ("final_heat_load_vs_drag", "final_net_heat_load_per_unit_span_J_m",
         "Final net heat load [J/m of span]", "Approximate Final Net Heat Load vs Drag"),
        ("peak_heat_flux_vs_drag", "peak_net_heat_flux_W_m2",
         "Peak net surface heat flux [W/m²]", "Approximate Peak Net Surface Heat Flux vs Drag"),
    )
    # These two tables were regenerated from the same saved thermal histories
    # immediately before this function is called. Their definitions stay shared.
    metrics = {}
    for filename in ("total_heat_load_comparison.csv", "peak_heat_flux_comparison.csv"):
        path = directory / filename
        if path.is_file():
            for row in read_csv(path):
                metrics.setdefault(row["design"], {}).update(row)
    records, pairing = [], []
    for name in cases:
        work = Path(runs_dir) / name
        reason = ""
        try:
            aero = read_aero(work)
            cfg = load_yaml(work / "_batch/config.yaml")
            thermal = read_json(work / "thermal_analysis/thermal_summary.json")
            drag = numeric(aero.get("drag_per_unit_span_N_m"))
            values = metrics.get(name, {})
            if not mechanical_complete(work, cfg):
                reason = "thermal_result_missing_or_stale"
            elif not aero_complete_for_report(cfg):
                reason = "aerodynamic_result_missing_or_stale"
            elif (not all(_thermal_primary_source(thermal).values())
                  or aero.get("source_sha256") != _thermal_primary_source(thermal)):
                reason = "aerodynamic_and_thermal_CFD_sources_differ"
            elif drag is None:
                reason = "sectional_drag_unavailable"
            elif any(numeric(values.get(field)) is None for _, field, _, _ in specifications):
                reason = "thermal_load_metrics_unavailable"
            else:
                checks = aero_report_fields(aero)
                records.append(dict(design=name, drag_per_unit_span_N_m=drag,
                    final_time_s=float(values["final_time_s"]),
                    final_net_heat_load_per_unit_span_J_m=float(values[specifications[0][1]]),
                    peak_net_heat_flux_W_m2=float(values[specifications[1][1]]),
                    time_of_peak_net_flux_s=float(values["time_of_peak_net_flux_s"]),
                    peak_net_surface_order_index=int(values["peak_net_surface_order_index"]),
                    force_stability_status=checks["force_stability_status"],
                    residual_status=checks["residual_status"],
                    force_checks_passed=force_screening_eligible({**checks,
                        "convergence_status": aero.get("convergence_status")})))
        except (OSError, ValueError, KeyError, TypeError):
            reason = "saved_pair_could_not_be_validated"
        pairing.append(dict(design=name, included=not bool(reason), reason=reason or "current_matched_CFD_source"))
    created = {}
    if pairing:
        created["thermal_drag_pairing_csv"] = str(write_csv(directory / "thermal_drag_pairing.csv", pairing))
        for row in pairing:
            if not row["included"]:
                print(f"[report] {row['design']}: heat/drag plots skipped ({row['reason']}); see {directory / 'thermal_drag_pairing.csv'}")
    # Remove only this function's artifacts if regeneration has no valid pairs.
    for stem, _, _, _ in specifications:
        for suffix in (".png", ".csv"):
            (directory / (stem + suffix)).unlink(missing_ok=True)
    if not records:
        return created
    plt = _plotting()
    for stem, field, ylabel, title in specifications:
        fig, ax = plt.subplots(figsize=(9, 6))
        for row in records:
            name = row["design"]
            ax.scatter(row["drag_per_unit_span_N_m"], row[field], color=colors[name],
                       marker="o" if row["force_checks_passed"] else "x", s=85)
            ax.annotate(name, (row["drag_per_unit_span_N_m"], row[field]),
                        xytext=(6, 6), textcoords="offset points", fontsize=9)
        ax.set(xlabel="Sectional drag [N/m of span]", ylabel=ylabel, title=title)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=.3)
        ax.margins(.15)
        if any(not row["force_checks_passed"] for row in records):
            fig.text(.5, .01, "× = drag stability or residual screening unverified/failed", ha="center", fontsize=9)
        created[stem + "_csv"] = str(write_csv(directory / (stem + ".csv"), records))
        created[stem + "_png"] = _save(fig, directory / (stem + ".png"), plt)
        print(f"[report] Saved heat/drag comparison: {created[stem + '_png']}")
    return created
