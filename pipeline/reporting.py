"""Data-first nose reports. Importing this module does not launch Ansys."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .trade_study import AERO_FIELDS, THERMAL_METRIC_FIELDS, read_aero


def load_yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def build_result_row(geometry_file, case_name, case_dir, status, elapsed_s, error=""):
    """Read persisted scientific values without rerunning or rewriting a solution."""
    case_dir = Path(case_dir)
    row = dict(geometry=Path(geometry_file).name, case=case_name, status=status,
               elapsed_s=round(float(elapsed_s), 3), error=error, boundary_layer_used="",
               max_heat_flux_W_m2="", mean_heat_flux_W_m2="", final_max_temperature_K="",
               final_mean_temperature_K="", thermal_nodes="", thermal_elements="")
    row.update({key: "" for key in (*AERO_FIELDS, *THERMAL_METRIC_FIELDS)})
    aero = read_aero(case_dir)
    row.update(aero_status=aero.get("status", "missing"), aero_error=aero.get("error", ""),
               coefficient_status=aero.get("coefficient_status", ""),
               convergence_status=aero.get("convergence_status", "not_verified"))
    if aero.get("status") == "complete":
        row.update({key: aero.get(key) for key in AERO_FIELDS})
    config_path = case_dir / "_batch/config.yaml"
    if config_path.is_file():
        cfg = load_yaml(config_path)
        row["boundary_layer_used"] = bool(cfg.get("mesh", {}).get("boundary_layer", {}).get("enabled", False))
    summary = read_json(case_dir / "thermal_analysis/thermal_summary.json")
    row["thermal_loading_mode"] = summary.get("loading_mode", "unknown")
    row["thermal_comparison_signature"] = json.dumps({
        key: summary.get(key) for key in ("material", "analysis", "loading_mode", "settings")}, sort_keys=True)
    row.update({key: summary.get("screening_metrics", {}).get(key, "") for key in THERMAL_METRIC_FIELDS})
    flux, temp = summary.get("applied_heat_flux_W_m2", {}), summary.get("final_temperature", {})
    mesh = summary.get("thermal_mesh", {})
    row.update(max_heat_flux_W_m2=flux.get("maximum", ""), mean_heat_flux_W_m2=flux.get("mean", ""),
               final_max_temperature_K=temp.get("maximum_K", ""), final_mean_temperature_K=temp.get("mean_K", ""),
               thermal_nodes=mesh.get("nodes", ""), thermal_elements=mesh.get("elements", ""))
    # A peak in an earlier result set can exceed the final temperature.
    history_path = case_dir / "thermal_analysis/temperature_history.csv"
    if history_path.is_file():
        maxima = [numeric(item.get("maximum_K")) for item in read_csv(history_path)]
        initial = numeric(summary.get("analysis", {}).get("initial_temperature_k"))
        valid = [value for value in maxima + [initial] if value is not None]
        if valid:
            row["peak_temperature_K"] = max(valid)
    return row


def save_batch_summary(output_dir, rows):
    output_dir = Path(output_dir)
    csv_path = write_csv(output_dir / "batch_summary.csv", rows)
    json_path = output_dir / "batch_summary.json"
    json_path.write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    return csv_path, json_path


def wall_coordinates(xy, nose_direction=(1.0, 0.0)):
    """Return original indices, physical arc length, and back→nose→back position.

    The straight rear closure is omitted only from the comparison profile. The
    full raw wall history remains intact. Each side's true arc length maps to
    half the normalized range; the nose is exactly 0.5. Temperatures always
    travel with their original coordinates, never shifted independently.
    """
    xy = np.asarray(xy, dtype=float)
    direction = np.asarray(nose_direction, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
        raise ValueError("Wall coordinates must be a finite ordered Nx2 polygon")
    if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("reporting.nose_direction must be a finite nonzero 2-D vector")
    direction /= np.linalg.norm(direction)
    axial = xy @ direction
    span = np.ptp(axial)
    if span <= 0:
        raise ValueError("Wall has zero axial extent")
    transverse = xy @ np.array([-direction[1], direction[0]])
    rear = np.flatnonzero(axial <= axial.min() + max(1e-10, span * 1e-5))
    if len(rear) < 2:
        raise ValueError("Cannot identify rear closure: set reporting.nose_direction toward the nose")
    start, end = int(rear[np.argmax(transverse[rear])]), int(rear[np.argmin(transverse[rear])])
    if start == end:
        raise ValueError("Rear closure has zero width")
    nose = int(np.argmax(axial))
    forward = [(start + offset) % len(xy) for offset in range((end - start) % len(xy) + 1)]
    reverse = [(start - offset) % len(xy) for offset in range((start - end) % len(xy) + 1)]
    indices = np.asarray(forward if nose in forward else reverse, dtype=int)
    if nose not in indices:
        raise ValueError("No rear-to-rear surface path contains the selected nose")
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy[indices], axis=0), axis=1))]
    pivot = int(np.flatnonzero(indices == nose)[0])
    if pivot == 0 or pivot == len(indices) - 1 or distance[pivot] <= 0 or distance[-1] <= distance[pivot]:
        raise ValueError("Nose must lie between rear endpoints")
    normalized = np.empty(len(indices))
    normalized[:pivot + 1] = 0.5 * distance[:pivot + 1] / distance[pivot]
    normalized[pivot:] = 0.5 + 0.5 * (distance[pivot:] - distance[pivot]) / (distance[-1] - distance[pivot])
    return indices, distance, normalized


def summarize_wall_history(records, design, nose_direction=(1, 0)):
    groups = defaultdict(list)
    for row in records:
        time = numeric(row.get("time_s"))
        if time is None or time < 0:
            raise ValueError("Wall history must contain finite nonnegative times")
        groups[time].append(row)
    if not groups:
        raise ValueError("Wall history is empty")
    history, final = [], None
    original_ids = original_xy = None
    for time in sorted(groups):
        sample = sorted(groups[time], key=lambda row: int(row["surface_order_index"]))
        ids = [int(row["surface_order_index"]) for row in sample]
        xy = np.array([[float(row["x_m"]), float(row["y_m"])] for row in sample])
        temperatures = np.array([float(row["wall_temperature_K"]) for row in sample])
        if len(set(ids)) != len(ids) or not np.isfinite(temperatures).all() or not np.isfinite(xy).all():
            raise ValueError("Duplicate wall indices or nonfinite wall data")
        if original_ids is not None and (ids != original_ids or not np.allclose(xy, original_xy, rtol=0, atol=1e-12)):
            raise ValueError("Wall coordinates or sample indices change between thermal result sets")
        original_ids, original_xy = ids, xy
        history.append(dict(design=design, time_s=time, min_wall_temperature_K=float(temperatures.min()),
                            max_wall_temperature_K=float(temperatures.max()),
                            mean_wall_temperature_K=float(temperatures.mean()), mean_definition="simple"))
        final = time, xy, temperatures
    time, xy, temperatures = final
    indices, distance, normalized = wall_coordinates(xy, nose_direction)
    profiles = [dict(design=design, final_time_s=time, surface_order_index=original_ids[int(i)],
                     x_m=float(xy[i, 0]), y_m=float(xy[i, 1]), s_m=float(s), s_norm=float(sn),
                     wall_temperature_K=float(temperatures[i]))
                for i, s, sn in zip(indices, distance, normalized)]
    peak = max(history, key=lambda row: row["max_wall_temperature_K"])
    summary = dict(design=design, mean_definition="simple", final_time_s=time,
                   global_peak_wall_temperature_K=peak["max_wall_temperature_K"],
                   time_of_global_peak_s=peak["time_s"], final_max_wall_temperature_K=float(temperatures.max()),
                   final_mean_wall_temperature_K=float(temperatures.mean()), nose_s_norm=0.5,
                   final_nose_wall_temperature_K=float(temperatures[int(indices[np.argmin(abs(normalized - 0.5))])]),
                   final_wall_sample_count_raw=len(temperatures))
    return history, profiles, summary


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path, plt, dpi=160):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return str(path)


def _wall_plots(history, profiles, directory, comparison=True, colors=None):
    plt = _plotting()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(row["design"] for row in history + profiles))
    colors = colors or {name: plt.get_cmap("tab10")(index % 10) for index, name in enumerate(names)}
    created = {}
    if history:
        fig, ax = plt.subplots(figsize=(10, 5))
        for name in names:
            values = sorted((r for r in history if r["design"] == name), key=lambda r: float(r["time_s"]))
            if not values:
                continue
            time = [float(r["time_s"]) for r in values]
            ax.plot(time, [float(r["max_wall_temperature_K"]) for r in values], color=colors[name], label=name)
            ax.plot(time, [float(r["mean_wall_temperature_K"]) for r in values], color=colors[name], linestyle="--")
        ax.set(xlabel="Time [s]", ylabel="Wall temperature [K]",
               title="Wall Temperature History" + (" Comparison" if comparison else "") + " (solid=max, dashed=simple mean)")
        ax.grid(alpha=0.3)
        ax.legend()
        created["wall_temperature_history_png"] = _save(fig, directory / "wall_temperature_history_comparison.png", plt)
    if profiles:
        fig, ax = plt.subplots(figsize=(10, 5))
        times = set()
        for name in names:
            values = sorted((r for r in profiles if r["design"] == name), key=lambda r: float(r["s_norm"]))
            if not values:
                continue
            times.update(float(r["final_time_s"]) for r in values)
            ax.plot([float(r["s_norm"]) for r in values], [float(r["wall_temperature_K"]) for r in values],
                    color=colors[name], label=name)
        time_label = f"t={next(iter(times)):g} s" if len(times) == 1 else "different final times; see CSV"
        ax.set(xlabel="Normalized distance along wall, back → nose → back, s*", ylabel="Wall temperature [K]",
               title=f"Final Wall Temperature ({time_label})", xlim=(0, 1))
        ax.grid(alpha=0.3)
        ax.legend()
        created["final_wall_temperature_normalized_png"] = _save(fig, directory / "final_wall_temperature_normalized_position.png", plt)
    return created


def create_comparison_outputs(runs_dir, rows, output_dir, reference_dir=None, case_names=None):
    """Rebuild every CSV/plot using saved data, with no CFD or thermal solve."""
    from .trade_study import create_trade_outputs
    runs_dir, output_dir = Path(runs_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories, profiles, summaries, warnings = [], [], [], []
    validated_raw_design = None
    design_order = [directory.name for directory in sorted(runs_dir.iterdir())
                    if (directory / "thermal_analysis/wall_temperature_history.csv").is_file()] if runs_dir.is_dir() else []
    if reference_dir is not None:
        for filename in ("wall_temperature_history_comparison.csv", "final_wall_temperature_normalized_position.csv"):
            reference_history = Path(reference_dir) / filename
            if reference_history.is_file():
                design_order.extend(row["design"] for row in read_csv(reference_history))
    plt = _plotting()
    colors = {name: plt.get_cmap("tab10")(index % 10)
              for index, name in enumerate(dict.fromkeys(design_order))}
    for directory in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        if case_names is not None and directory.name not in case_names:continue
        raw = directory / "thermal_analysis/wall_temperature_history.csv"
        if not directory.is_dir() or not raw.is_file():
            continue
        try:
            cfg_path = directory / "_batch/config.yaml"
            cfg = load_yaml(cfg_path) if cfg_path.is_file() else {}
            history, profile, summary = summarize_wall_history(read_csv(raw), directory.name,
                cfg.get("reporting", {}).get("nose_direction", [1, 0]))
            histories.extend(history)
            profiles.extend(profile)
            summaries.append(summary)
            report_dir = directory / "reports"
            write_csv(report_dir / "wall_temperature_history_summary.csv", history)
            write_csv(report_dir / "final_wall_temperature.csv", profile)
            _wall_plots(history, profile, report_dir, comparison=False, colors=colors)
            _source_wall_plots(read_csv(raw), report_dir)
            saved_arrays = directory / "thermal_analysis/thermal_snapshots.npz"
            if saved_arrays.is_file():
                with np.load(saved_arrays, allow_pickle=False) as arrays:
                    snapshots = [dict(time_s=float(t), temperature_K=values)
                                 for t, values in zip(arrays["time_s"], arrays["temperatures_K"])]
                    node_history = [dict(time_s=float(s["time_s"]), minimum_K=float(np.min(s["temperature_K"])),
                                         mean_K=float(np.mean(s["temperature_K"])), maximum_K=float(np.max(s["temperature_K"])))
                                    for s in snapshots]
                    summary = read_json(directory / "thermal_analysis/thermal_summary.json")
                    create_thermal_visualizations(directory / "thermal_analysis", arrays["boundary_xy_m"],
                        arrays["boundary_flux_W_m2"] if "boundary_flux_W_m2" in arrays else np.zeros(len(arrays["boundary_xy_m"])), arrays["coordinates_m"], node_history, snapshots,
                        summary.get("visualization_settings", {"make_gif": False}))
        except (ValueError, KeyError, TypeError) as exc:
            warnings.append(f"{directory.name}: {exc}")
    if reference_dir is not None:
        reference_dir = Path(reference_dir)
        for filename, destination in (("wall_temperature_history_comparison.csv", histories),
                                      ("final_wall_temperature_normalized_position.csv", profiles),
                                      ("temperature_comparison_summary.csv", summaries)):
            path = reference_dir / filename
            if path.is_file():
                imported = read_csv(path)
                # Imported normalized positions are authoritative; do not roll,
                # stretch, interpolate, or recompute their temperature means.
                for row in imported:
                    for key in tuple(row):
                        if key not in ("design", "source_csv", "mean_definition"):
                            value = numeric(row[key])
                            if value is not None:
                                row[key] = value
                    if destination is histories:
                        row["mean_definition"] = "simple"
                known = {row["design"] for row in destination}
                destination.extend(row for row in imported if row["design"] not in known)
        raw_path = reference_dir / "wall_temperature_history.csv"
        if raw_path.is_file():
            raw = read_csv(raw_path)
            # A label-free historical wall file is assigned to a design only
            # after matching ALL saved times, maxima and simple wall means.
            try:
                raw_history, raw_profile, raw_summary = summarize_wall_history(raw, "unlabeled", [1, 0])
                matches = []
                for name in dict.fromkeys(row["design"] for row in histories):
                    candidate = sorted([row for row in histories if row["design"] == name], key=lambda row: float(row["time_s"]))
                    if len(candidate) == len(raw_history) and all(
                        math.isclose(float(left[key]), float(right[key]), rel_tol=1e-9, abs_tol=1e-8)
                        for left, right in zip(candidate, raw_history)
                        for key in ("time_s", "max_wall_temperature_K", "mean_wall_temperature_K")):
                        matches.append(name)
                if len(matches) == 1:
                    validated_raw_design = matches[0]
                    directory = output_dir / "cases" / validated_raw_design
                    write_csv(directory / "wall_temperature_history.csv", raw)
                    _source_wall_plots(raw, directory)
                else:
                    warnings.append("Unlabeled raw wall history has no unique numerical match to a combined design")
            except (ValueError, KeyError, TypeError) as exc:
                warnings.append(f"Historical raw wall history: {exc}")
    # Imported combined data get the same per-nose tables and plots as new
    # simulation data, without inventing missing physical coordinates or loads.
    for name in dict.fromkeys(row["design"] for row in histories + profiles):
        case_history = [row for row in histories if row["design"] == name]
        case_profile = [row for row in profiles if row["design"] == name]
        directory = output_dir / "cases" / name
        if case_history:
            write_csv(directory / "wall_temperature_history_summary.csv", case_history)
        if case_profile:
            write_csv(directory / "final_wall_temperature.csv", case_profile)
        _wall_plots(case_history, case_profile, directory, comparison=False, colors=colors)
    created = {}
    for filename, records in (("wall_temperature_history_comparison.csv", histories),
                              ("final_wall_temperature_normalized_position.csv", profiles),
                              ("temperature_comparison_summary.csv", summaries)):
        if records:
            created[filename[:-4]] = str(write_csv(output_dir / filename, records))
    created.update(_wall_plots(histories, profiles, output_dir, colors=colors))
    created.update(create_trade_outputs(runs_dir, rows, output_dir))
    manifest = dict(status="complete" if not warnings else "partial", warnings=warnings,
                    wall_mean_definition="simple arithmetic mean of saved wall samples; not area weighted",
                    normalization="rear→nose→rear; physical arc length on each side maps to half-range",
                    reference_data="Imported historical data are report fixtures, not new simulation results.",
                    validated_unlabeled_wall_design=validated_raw_design,
                    outputs=created)
    manifest_path = output_dir / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    created["report_manifest"] = str(manifest_path)
    return created


def _source_wall_plots(records, directory):
    """Plot only measured/saved wall quantities; a wall is not a volume mesh."""
    plt = _plotting()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    time = max(float(row["time_s"]) for row in records)
    rows = sorted((row for row in records if float(row["time_s"]) == time),
                  key=lambda row: int(row["surface_order_index"]))
    xy = np.array([[float(row["x_m"]), float(row["y_m"])] for row in rows])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xy[:, 0], xy[:, 1], label="Saved CFD wall coordinates")
    ax.scatter(xy[:, 0], xy[:, 1], s=10)
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Saved Wall Boundary")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    _save(fig, directory / "wall_boundary.png", plt)
    flux_key = next((key for key in ("net_surface_heat_flux_into_wall_W_m2", "convective_heat_flux_into_wall_W_m2",
                                    "applied_heat_flux_into_wall_W_m2") if rows and key in rows[0]), None)
    if flux_key:
        q = np.array([float(row[flux_key]) for row in rows])
        fig, ax = plt.subplots(figsize=(9, 4))
        points = ax.scatter(xy[:, 0], xy[:, 1], c=q, s=20)
        ax.plot(xy[:, 0], xy[:, 1], linewidth=0.5, color="black")
        fig.colorbar(points, ax=ax, label=flux_key.replace("_", " "))
        ax.set(xlabel="x [m]", ylabel="y [m]", title=f"Saved Surface Heat Flux (t={time:g} s)")
        ax.set_aspect("equal", adjustable="box")
        _save(fig, directory / "final_wall_heat_flux_map.png", plt)


def create_thermal_visualizations(output_dir, boundary_xy, boundary_q, xyz, history, snapshots, viz_cfg):
    """Render MAPDL arrays plus wall mapping; FE results remain authoritative."""
    if not viz_cfg.get("enabled", True):
        return {}
    if not snapshots:
        raise ValueError("No thermal snapshots are available")
    import matplotlib.tri as mtri
    from matplotlib.path import Path as PolygonPath
    plt = _plotting()
    output_dir = Path(output_dir) / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    xy, xyz, q = np.asarray(boundary_xy), np.asarray(xyz), np.asarray(boundary_q)
    dpi = max(72, int(viz_cfg.get("dpi", 160)))
    final = np.asarray(snapshots[-1]["temperature_K"])
    time = snapshots[-1]["time_s"]
    tri = mtri.Triangulation(xyz[:, 0], xyz[:, 1])
    centers = xyz[tri.triangles, :2].mean(axis=1)
    tri.set_mask(~PolygonPath(np.vstack([xy, xy[0]])).contains_points(centers))
    # This is a masked plotting triangulation, not the MAPDL element connectivity.
    values = np.concatenate([np.asarray(s["temperature_K"]) for s in snapshots])
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        lo, hi = lo - 1, hi + 1
    levels = np.linspace(lo, hi, 24)
    created = {}

    def axes(title):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set(xlabel="x [m]", ylabel="y [m]", title=title)
        ax.set_aspect("equal", adjustable="box")
        return fig, ax

    def outline(ax):
        closed = np.vstack([xy, xy[0]])
        ax.plot(closed[:, 0], closed[:, 1], linewidth=0.7, color="black")

    def field(snapshot, filename):
        fig, ax = axes(f"2-D Temperature Field (t={snapshot['time_s']:g} s)")
        field = ax.tricontourf(tri, np.asarray(snapshot["temperature_K"]), levels=levels)
        fig.colorbar(field, ax=ax, label="Temperature [K]")
        outline(ax)
        return _save(fig, output_dir / filename, plt, dpi)

    if viz_cfg.get("save_final_temperature_image", True):
        created["final_temperature_field_png"] = field(snapshots[-1], "final_temperature_field.png")
        fig, ax = axes(f"Final Thermal Nodes (t={time:g} s)")
        points = ax.scatter(xyz[:, 0], xyz[:, 1], c=final, s=8)
        fig.colorbar(points, ax=ax, label="Temperature [K]")
        created["final_thermal_nodes_png"] = _save(fig, output_dir / "final_thermal_nodes.png", plt, dpi)
    fig, ax = axes("CFD Boundary Points vs Thermal Wall")
    ax.scatter(xy[:, 0], xy[:, 1], s=12, label="CFD boundary points")
    ax.plot(xy[:, 0], xy[:, 1], label="Reconstructed thermal boundary")
    ax.legend()
    created["boundary_mapping_png"] = _save(fig, output_dir / "boundary_mapping.png", plt, dpi)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    nearest = np.concatenate([np.argmin(((xy[i:i+64, None, :] - xyz[None, :, :2]) ** 2).sum(axis=2), axis=1)
                              for i in range(0, len(xy), 64)])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(distance, final[nearest])
    ax.set(xlabel="Distance along supplied wall order [m]", ylabel="Wall temperature [K]",
           title=f"Final Wall Temperature (t={time:g} s)")
    ax.grid(alpha=0.3)
    created["final_wall_temperature_png"] = _save(fig, output_dir / "final_wall_temperature.png", plt, dpi)
    if viz_cfg.get("save_heat_flux_image", True):
        fig, ax = axes("Applied Cold-Wall Aerothermal Heat Flux")
        points = ax.scatter(xy[:, 0], xy[:, 1], c=q, s=18)
        outline(ax)
        fig.colorbar(points, ax=ax, label="Heat flux into wall [W/m²]")
        created["applied_heat_flux_map_png"] = _save(fig, output_dir / "applied_heat_flux_map.png", plt, dpi)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(distance, q)
        ax.set(xlabel="Distance along supplied wall order [m]", ylabel="Heat flux into wall [W/m²]",
               title="Applied Cold-Wall Heat Flux vs Position")
        ax.grid(alpha=0.3)
        created["applied_heat_flux_position_png"] = _save(fig, output_dir / "applied_heat_flux_position.png", plt, dpi)
    if history and viz_cfg.get("save_temperature_history_image", True):
        fig, ax = plt.subplots(figsize=(9, 5))
        for key, label in (("minimum_K", "Minimum"), ("mean_K", "Nodal mean"), ("maximum_K", "Maximum")):
            ax.plot([r["time_s"] for r in history], [r[key] for r in history], label=label)
        ax.set(xlabel="Time [s]", ylabel="Temperature [K]", title="Solid Temperature History")
        ax.grid(alpha=0.3)
        ax.legend()
        created["temperature_history_png"] = _save(fig, output_dir / "temperature_history.png", plt, dpi)
    if viz_cfg.get("make_gif", True) and len(snapshots) >= 2:
        from PIL import Image
        count = min(len(snapshots), max(2, int(viz_cfg.get("max_animation_frames", 60))))
        frames, paths = [], []
        try:
            for number, index in enumerate(np.unique(np.linspace(0, len(snapshots) - 1, count, dtype=int))):
                path = Path(field(snapshots[index], f"frame_{number:04d}.png"))
                paths.append(path)
                with Image.open(path) as source:
                    frames.append(source.convert("RGB").copy())
            target = output_dir / "nose_heating_animation.gif"
            frames[0].save(target, save_all=True, append_images=frames[1:], loop=0,
                           duration=max(20, round(1000 / max(1, int(viz_cfg.get("fps", 8))))))
            created["nose_heating_animation_gif"] = str(target)
        finally:
            for frame in frames:
                frame.close()
            if not viz_cfg.get("save_frame_pngs", False):
                for path in paths:
                    path.unlink(missing_ok=True)
    return created
