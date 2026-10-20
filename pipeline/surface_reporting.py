"""Spatial and physical-time plots of saved surface loads; no Ansys execution."""
from pathlib import Path
from textwrap import fill

import numpy as np

from .reporting import _plotting, _save, numeric, plot_thermal_boundary, thermal_segment_flags, wall_coordinates, write_csv


def surface_quantity_outputs(kind, fields, cases, summarized, directory, colors):
    """Keep signed convection, radiation and net quantities separate.

    Local heat load is saved cumulative energy per area, never flux multiplied
    by the final time. Maps show boundary samples, not an interior heat field.
    """
    directory = Path(directory)
    plt = _plotting()
    label = "Heat Flux" if kind == "flux" else "Accumulated Heat Load"
    unit = "W/m²" if kind == "flux" else "J/m²"
    stem = "surface_heat_flux" if kind == "flux" else "surface_heat_load"
    names = [name for name in cases if any(row["design"] == name for row in summarized)]
    created = {}
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (field, component, _) in zip(axes, fields):
        for name in names:
            records = sorted((row for row in summarized if row["design"] == name), key=lambda row: row["time_s"])
            times = [row["time_s"] for row in records]
            ax.plot(times, [row["maximum_" + field] for row in records], color=colors[name], label=name)
            ax.plot(times, [row["mean_" + field] for row in records], color=colors[name], linestyle="--")
        ax.set(ylabel=f"{component}\n[{unit}]")
        ax.grid(alpha=.3)
    axes[0].set_title(f"Approximate {label} History (solid=max, dashed=simple analyzed-wall mean)")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Physical time [s]")
    created[stem + "_max_mean_history_png"] = _save(fig, directory / (stem + "_max_mean_history.png"), plt)

    final, profiles = {}, []
    for name in names:
        records = cases[name][kind]
        time = max(float(row["time_s"]) for row in records)
        records = sorted((row for row in records if float(row["time_s"]) == time),
                         key=lambda row: int(row["surface_order_index"]))
        xy = np.asarray([[float(row["x_m"]), float(row["y_m"])] for row in records])
        indices, distance, normalized = wall_coordinates(xy, cases[name].get("nose_direction", [1., 0.]))
        final[name] = time, xy, records
        for index, s, sn in zip(indices, distance, normalized):
            row = records[index]
            profiles.append(dict(design=name, final_time_s=time, surface_order_index=int(row["surface_order_index"]),
                                 x_m=float(row["x_m"]), y_m=float(row["y_m"]), s_m=float(s), s_norm=float(sn),
                                 **({"thermal_segment_active": int(thermal_segment_flags([row])[0])}
                                    if "thermal_segment_active" in row else {}),
                                 **{field: float(row[field]) for field, _, _ in fields}))
    created["final_" + stem + "_profile_csv"] = str(write_csv(directory / ("final_" + stem + "_profile.csv"), profiles))
    for coordinate, suffix, xlabel in (
        ("s_norm", "normalized_position", "Normalized wall position, back → nose → back, s*"),
        ("s_m", "position", "Distance along wall, back → nose → back [m]"),
    ):
        # The normalized profile is the common comparison coordinate. Physical
        # arc length is additionally useful for each individual nose.
        if coordinate == "s_m" and len(names) != 1:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for ax, (field, component, _) in zip(axes, fields):
            for name in names:
                records = [row for row in profiles if row["design"] == name]
                ax.plot([row[coordinate] for row in records], [row[field] for row in records],
                        color=colors[name], label=f"{name} (t={final[name][0]:g} s)")
            ax.set(ylabel=f"{component}\n[{unit}]")
            ax.grid(alpha=.3)
        axes[0].set_title(f"Approximate Final Surface {label}")
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel(xlabel)
        if coordinate == "s_norm":axes[-1].set_xlim(0, 1)
        key = "final_" + stem + "_" + suffix + "_png"
        created[key] = _save(fig, directory / ("final_" + stem + "_" + suffix + ".png"), plt)

    fig, axes = plt.subplots(len(names), 3, figsize=(15, max(3, 2.7 * len(names))), squeeze=False,
                             layout="constrained")
    for column, (field, component, _) in enumerate(fields):
        values = [float(row[field]) for _, _, records in final.values() for row in records]
        vmin, vmax = min(values), max(values)
        for number, name in enumerate(names):
            time, xy, records = final[name]
            ax = axes[number, column]
            plot_thermal_boundary(ax, records, color="black", linewidth=.6)
            points = ax.scatter(xy[:, 0], xy[:, 1], c=[float(row[field]) for row in records],
                                s=18, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set(xlabel="x [m]", ylabel="y [m]", title=f"{fill(name, 32)} (t={time:g} s)\n{component}")
            ax.title.set_fontsize(10)
            ax.set_aspect("equal", adjustable="box")
        fig.colorbar(points, ax=axes[:, column].tolist(), label=unit, shrink=.8)
    scoped = any((thermal_segment_flags(records) is not None and not thermal_segment_flags(records).all())
                 for _, _, records in final.values())
    scope_label = "analyzed boundary only; insulated rear omitted" if scoped else "boundary samples"
    fig.suptitle(f"Approximate Final Surface {label} — {scope_label}; common scale within each column")
    path = directory / ("final_" + stem + "_maps.png")
    # Constrained layout preserves shared colorbars on thin planar profiles.
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created["final_" + stem + "_maps_png"] = str(path)
    return created


def peak_temperature_outputs(histories, directory, colors):
    """Peak over all saved solid nodes and physical times, including t=0.

    Wall-only histories are intentionally not substituted for an all-solid
    maximum. Neither the final temperature nor a mean is used as the peak.
    """
    if not histories:return {}
    rows = []
    for name, records in histories.items():
        values = [(numeric(row.get("time_s")), numeric(row.get("maximum_K"))) for row in records]
        if not values or any(t is None or t < 0 or value is None or value <= 0 for t, value in values):
            raise ValueError("Peak temperature requires finite positive solid temperatures and physical times")
        values.sort()
        if any(right[0] <= left[0] for left, right in zip(values, values[1:])):
            raise ValueError("Peak temperature history has duplicate physical times")
        time, peak = max(values, key=lambda item: item[1])
        rows.append(dict(design=name, peak_temperature_K=peak, time_of_peak_s=time,
                         final_time_s=values[-1][0], sampling="all saved solid nodes and physical result sets"))
    directory = Path(directory)
    plt = _plotting()
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(rows)), 5))
    bars = ax.bar(np.arange(len(rows)), [row["peak_temperature_K"] for row in rows],
                  color=[colors[row["design"]] for row in rows])
    ax.set_xticks(np.arange(len(rows)), [row["design"] for row in rows], rotation=25, ha="right")
    ax.bar_label(bars, labels=[f"{row['peak_temperature_K']:,.1f} K" for row in rows], padding=4)
    ax.set(ylabel="Peak solid temperature [K]", title="Maximum Temperature During Each Exposure (all saved solid nodes)")
    ax.set_ylim(0, max(row["peak_temperature_K"] for row in rows) * 1.15)
    ax.grid(axis="y", alpha=.3)
    ax.set_axisbelow(True)
    return dict(max_temperature_comparison_csv=str(write_csv(directory / "max_temperature_comparison.csv", rows)),
                max_temperature_comparison_png=_save(fig, directory / "max_temperature_comparison.png", plt))
