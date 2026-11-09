"""Spatial comparison of saved surface heat fluxes without a new solver run."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .reporting import (
    _plotting, _save, numeric, thermal_segment_flags, wall_coordinates, write_csv,
)

COMPONENTS = (
    ("convective", "convective_heat_flux_into_solid_W_m2", "Convection into solid"),
    ("radiative", "radiative_heat_flux_out_W_m2", "Radiation out"),
    ("net", "net_heat_flux_into_solid_W_m2", "Net into solid"),
)


def _profile_case(name, source):
    groups = defaultdict(list)
    for original in source.get("flux", []):
        time = numeric(original.get("time_s"))
        index = numeric(original.get("surface_order_index"))
        if time is None or time < 0 or index is None or index < 0 or not index.is_integer():
            raise ValueError("Spatial heat flux requires finite nonnegative times and integer surface indices")
        row = dict(original, time_s=time, surface_order_index=int(index))
        for field in ("x_m", "y_m", *(field for _, field, _ in COMPONENTS)):
            value = numeric(row.get(field))
            if value is None:
                raise ValueError(f"Spatial heat flux requires finite {field} samples")
            row[field] = value
        groups[time].append(row)
    if not groups:
        raise ValueError("No saved flux samples for the spatial comparison")
    times = sorted(groups)
    baseline_ids = baseline_xy = baseline_flags = None
    snapshots = []
    for time in times:
        records = sorted(groups[time], key=lambda row: row["surface_order_index"])
        ids = [row["surface_order_index"] for row in records]
        xy = np.asarray([[row["x_m"], row["y_m"]] for row in records])
        flags = thermal_segment_flags(records)
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate surface sample at one physical time")
        if baseline_ids is not None:
            if ids != baseline_ids or not np.allclose(xy, baseline_xy, rtol=0, atol=1e-12):
                raise ValueError("Surface sample indices or coordinates change between saved times")
            if ((flags is None) != (baseline_flags is None)
                    or flags is not None and not np.array_equal(flags, baseline_flags)):
                raise ValueError("Surface activity flags change between saved times")
        else:
            baseline_ids, baseline_xy, baseline_flags = ids, xy, flags
        snapshots.append(records)
    indices, distance, normalized = wall_coordinates(
        baseline_xy, np.asarray(source.get("nose_direction", [1., 0.]), dtype=float).copy())
    if baseline_flags is not None:
        count = len(baseline_ids)
        for first, second in zip(indices[:-1], indices[1:]):
            edge = int(first if second == (first + 1) % count else second)
            if not baseline_flags[edge]:
                raise ValueError("An excluded segment lies within the forward spatial profile; no curve was drawn across it")
    initial, final, envelope = [], [], []
    for index, arc, sn in zip(indices, distance, normalized):
        identity = dict(design=name, surface_order_index=baseline_ids[index],
                        x_m=float(baseline_xy[index, 0]), y_m=float(baseline_xy[index, 1]),
                        s_m=float(arc), s_norm=float(sn))
        if baseline_flags is not None:
            identity["thermal_segment_active"] = int(baseline_flags[index])
        for snapshot, destination in ((snapshots[0], initial), (snapshots[-1], final)):
            destination.append(dict(identity, time_s=snapshot[index]["time_s"],
                                    **{field: snapshot[index][field] for _, field, _ in COMPONENTS}))
        peak = dict(identity, first_time_s=times[0], final_time_s=times[-1])
        for component, field, _ in COMPONENTS:
            maximum = max((snapshot[index] for snapshot in snapshots), key=lambda row: row[field])
            peak[f"peak_{component}_heat_flux_W_m2"] = maximum[field]
            peak[f"time_of_peak_{component}_flux_s"] = maximum["time_s"]
        envelope.append(peak)
    return initial, final, envelope


def spatial_flux_outputs(cases, directory, colors):
    """Plot each geometry's own ordered samples in a common surface coordinate.

    Return ``files`` and per-case ``excluded`` entries. Failure to identify a
    valid spatial profile does not invalidate an otherwise computable peak bar.
    No heat flux is inferred from temperature, and no values are interpolated
    from one geometry onto another. Envelopes are maxima at each saved location,
    not the flux distribution at a single physical instant.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("initial_heat_flux_spatial_comparison", "Initial Saved Surface Heat Flux", 0),
        ("final_heat_flux_spatial_comparison", "Final Saved Surface Heat Flux", 1),
        ("peak_heat_flux_spatial_envelope", "Maximum Heat Flux at Each Wall Location Over Saved Time", 2),
    )
    # Clear only this helper's generated files so excluded cases cannot leave
    # curves from an older invocation behind.
    for stem, _, _ in specifications:
        for extension in (".png", ".csv"):
            (directory / (stem + extension)).unlink(missing_ok=True)
    profiles, excluded = {}, []
    for name, source in cases.items():
        try:
            profiles[name] = _profile_case(name, source)
        except (TypeError, ValueError, KeyError) as exc:
            excluded.append(dict(case=name, reason=str(exc)))
    if not profiles:
        return dict(files={}, excluded=excluded)
    plt, files = _plotting(), {}
    for stem, title, number in specifications:
        combined = [row for profile in profiles.values() for row in profile[number]]
        files[stem + "_csv"] = str(write_csv(directory / (stem + ".csv"), combined))
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for axis, (component, field, label) in zip(axes, COMPONENTS):
            for name, profile in profiles.items():
                rows = profile[number]
                column = f"peak_{component}_heat_flux_W_m2" if number == 2 else field
                legend = name if number == 2 else f"{name} (t={rows[0]['time_s']:g} s)"
                axis.plot([row["s_norm"] for row in rows], [row[column] for row in rows],
                          color=colors.get(name), label=legend)
            axis.axvline(.5, color="grey", linestyle=":", linewidth=.7)
            axis.set_ylabel(label + "\n[W/m²]")
            axis.grid(alpha=.3)
        axes[0].set_title(title)
        axes[0].legend(fontsize=9)
        axes[-1].set(xlabel="Normalized wall distance, back → nose → back, s*", xlim=(0, 1))
        if number == 2:
            fig.text(.5, .005, "Pointwise time maxima; different locations can peak at different times. Not a single physical snapshot.",
                     ha="center", fontsize=8)
        files[stem + "_png"] = _save(fig, directory / (stem + ".png"), plt)
    return dict(files=files, excluded=excluded)
