"""Internal-temperature screening of saved physical-time probe histories.

The user-specified temperature is a screening criterion, not a constitutive
failure model. Crossing estimates retain their actual saved-time brackets.
"""
from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np


def first_threshold_crossing(times_s, temperatures_K, limit_K=1800.):
    """Return the first upward threshold crossing and its saved data bracket.

    Linear interpolation estimates only the crossing between recorded samples;
    it does not reconstruct unrecorded temperature peaks or prove failure.
    """
    times, temperatures = np.asarray(times_s, dtype=float), np.asarray(temperatures_K, dtype=float)
    limit = float(limit_K)
    if (times.ndim != 1 or len(times) < 2 or temperatures.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(temperatures).all()
            or (times < 0).any() or abs(times[0]) > 1e-12 or (np.diff(times) <= 0).any()
            or (temperatures <= 0).any() or not np.isfinite(limit) or limit <= 0):
        raise ValueError("Screening requires increasing physical times from zero, positive absolute temperatures and a positive limit")
    peak_index = int(np.argmax(temperatures))
    result = dict(status="not_reached_within_duration", time_to_limit_s=None,
                  lower_time_s=None, upper_time_s=None,
                  lower_temperature_K=None, upper_temperature_K=None,
                  duration_s=float(times[-1]), limit_K=limit,
                  peak_temperature_K=float(temperatures[peak_index]),
                  time_of_peak_s=float(times[peak_index]), final_temperature_K=float(temperatures[-1]),
                  crossing_method="linear interpolation between saved physical-time samples",
                  interpretation="User-defined internal-temperature screening; not a structural failure prediction")
    if temperatures[0] >= limit:
        result.update(status="initially_at_or_above_limit", time_to_limit_s=0.,
                      lower_time_s=0., upper_time_s=0.,
                      lower_temperature_K=float(temperatures[0]), upper_temperature_K=float(temperatures[0]),
                      crossing_method="initial saved state already at or above screening limit")
        return result
    reached = np.flatnonzero(temperatures >= limit)
    if len(reached):
        upper = int(reached[0])
        lower = upper - 1
        fraction = (limit - temperatures[lower]) / (temperatures[upper] - temperatures[lower])
        result.update(status="reached", time_to_limit_s=float(times[lower] + fraction * (times[upper] - times[lower])),
                      lower_time_s=float(times[lower]), upper_time_s=float(times[upper]),
                      lower_temperature_K=float(temperatures[lower]), upper_temperature_K=float(temperatures[upper]))
    return result


def screening_bar_outputs(rows, output_dir, limit_K=1800., reference_time_s=None):
    """Compare internal peaks and distinguish crossing times from lower bounds.

    All rows must identify the same physical depth. Parent code must additionally
    group compatible material, depth convention, flight and boundary conditions.
    A hatched time bar ends at its simulation horizon and is labeled as a lower
    bound; the CSV never replaces an unreached crossing with zero or the horizon.
    """
    from .reporting import _plotting, _save, numeric, write_csv

    if not rows:return {}
    limit = numeric(limit_K)
    if limit is None or limit <= 0:raise ValueError("Internal screening limit must be a positive kelvin temperature")
    reference = numeric(reference_time_s) if reference_time_s is not None else None
    if reference_time_s is not None and (reference is None or reference < 0):
        raise ValueError("Reference time must be finite and nonnegative")
    records, names = [], set()
    for original in rows:
        row = dict(original)
        name = row.get("design")
        status = row.get("status")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Internal screening bars require unique nose names")
        names.add(name)
        for field in ("peak_temperature_K", "final_temperature_K", "duration_s", "depth_m"):
            value = numeric(row.get(field))
            if value is None or value <= 0:raise ValueError(f"Internal screening bars require positive finite {field}")
            row[field] = value
        if row["final_temperature_K"] > row["peak_temperature_K"] + 1e-8:
            raise ValueError("A final internal temperature cannot exceed the recorded peak")
        if status == "not_reached_within_duration":
            if row.get("time_to_limit_s") is not None or row["peak_temperature_K"] >= limit:
                raise ValueError("An unreached screening limit must retain a missing crossing time and a sub-limit peak")
            row.update(time_to_limit_s=None, time_to_limit_lower_bound_s=row["duration_s"],
                       time_bar_height_s=row["duration_s"], time_bar_is_lower_bound=True)
        elif status in {"reached", "initially_at_or_above_limit"}:
            crossing = numeric(row.get("time_to_limit_s"))
            if crossing is None or not 0 <= crossing <= row["duration_s"] or row["peak_temperature_K"] < limit:
                raise ValueError("Invalid recorded time to the internal screening limit")
            if status == "initially_at_or_above_limit" and crossing != 0:
                raise ValueError("An initially exceeded screening limit must have crossing time zero")
            row.update(time_to_limit_s=crossing, time_to_limit_lower_bound_s=None,
                       time_bar_height_s=crossing, time_bar_is_lower_bound=False)
        else:
            raise ValueError(f"Unknown internal-temperature screening status: {status!r}")
        row["limit_K"] = limit
        records.append(row)
    depths = np.asarray([row["depth_m"] for row in records])
    if not np.allclose(depths, depths[0], rtol=0, atol=1e-12):
        raise ValueError("Internal screening bars must compare the same depth")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    plt = _plotting()
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(records))]
    positions = np.arange(len(records))
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(records)), 5))
    bars = ax.bar(positions, [row["peak_temperature_K"] for row in records], color=colors)
    ax.bar_label(bars, labels=[f"{row['peak_temperature_K']:,.1f} K" for row in records], padding=4)
    ax.axhline(limit, color="firebrick", linestyle="--", label=f"Screening limit: {limit:g} K")
    ax.set_xticks(positions, [row["design"] for row in records], rotation=25, ha="right")
    ax.set(ylabel="Peak internal temperature [K]", xlabel="Nose geometry",
           title=f"Peak Internal Temperature at {depths[0] * 1000:g} mm Depth")
    ax.set_ylim(0, max(limit, max(row["peak_temperature_K"] for row in records)) * 1.15)
    ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True); ax.legend(fontsize=9)
    extended = [row["design"] for row in records if row.get("material_extrapolation_used")]
    if extended:
        fig.text(.5, .01, textwrap.fill("Endpoint material properties used outside table range: " + ", ".join(extended),
                 width=int(fig.get_figwidth() * 15)), ha="center", va="bottom", fontsize=8)
    peak_stem = "peak_internal_temperature_comparison"
    created = {peak_stem + "_csv": str(write_csv(directory / (peak_stem + ".csv"), records)),
               peak_stem + "_png": _save(fig, directory / (peak_stem + ".png"), plt, rect=(0, .1 if extended else 0, 1, 1))}
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(records)), 5.5))
    bars = ax.bar(positions, [row["time_bar_height_s"] for row in records], color=colors)
    for bar, row in zip(bars, records):
        if row["time_bar_is_lower_bound"]:bar.set_hatch("///")
    ax.bar_label(bars, labels=[f">{row['duration_s']:g} s" if row["time_bar_is_lower_bound"]
                              else f"{row['time_to_limit_s']:.2f} s" for row in records], padding=4)
    if reference is not None:
        ax.axhline(reference, color="grey", linestyle=":", label=f"User reference: {reference:g} s")
        ax.legend(fontsize=9)
    ax.set_xticks(positions, [row["design"] for row in records], rotation=25, ha="right")
    ax.set(ylabel="Time to internal screening limit [s]", xlabel="Nose geometry",
           title=f"Time to {limit:g} K at {depths[0] * 1000:g} mm Depth")
    ax.set_ylim(0, max(1., reference or 0., max(row["time_bar_height_s"] for row in records)) * 1.18)
    ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
    fig.text(.5, .005, "Hatched bar = limit not reached within saved duration; bar height is a lower bound, not a crossing.",
             ha="center", fontsize=8)
    if extended:
        fig.text(.5, .035, textwrap.fill("Endpoint material properties used outside table range: " + ", ".join(extended),
                 width=int(fig.get_figwidth() * 15)), ha="center", va="bottom", fontsize=8)
    time_stem = "time_to_internal_limit_comparison"
    created.update({time_stem + "_csv": str(write_csv(directory / (time_stem + ".csv"), records)),
                    time_stem + "_png": _save(fig, directory / (time_stem + ".png"), plt, rect=(0, .15 if extended else .07, 1, 1))})
    return created
