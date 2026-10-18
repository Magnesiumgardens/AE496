"""Measured sectional forces versus Fluent iteration, with a separate stability screen.

The native report file records every iteration. The public CSV samples the
configured interval and always includes the independently measured final
iteration. Neither the requested iteration budget nor temperature-history
timestamps are used to manufacture force samples.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shlex
from statistics import mean

from .temperature_export import atomic_csv, temperature_conditions
from .utils import file_sha256


CSV_NAME = "solver_force_history.csv"
RAW_NAME = "solver_force_history.out"
FORCE_COLUMNS = ("drag_per_unit_span_N_m", "lift_per_unit_span_N_m")
DEFAULTS = dict(enabled=True, sample_every_iterations=25, recent_samples=5,
                min_samples=5, relative_tolerance=0.01,
                absolute_drag_tolerance_N_m=1e-6, require_stable=True)


def settings(cfg):
    """Validate sampling and screening settings without altering solver controls."""
    requested = cfg.get("aero", {}).get("force_history", {})
    if not isinstance(requested, dict):
        raise ValueError("aero.force_history must be a mapping")
    unknown = set(requested) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown aero.force_history setting(s): " + ", ".join(sorted(unknown)))
    result = {**DEFAULTS, **requested}
    for key in ("enabled", "require_stable"):
        if not isinstance(result[key], bool):
            raise ValueError(f"aero.force_history.{key} must be true or false")
    for key in ("sample_every_iterations", "recent_samples", "min_samples"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"aero.force_history.{key} must be a positive integer")
        if int(value) != value or value < (2 if key == "recent_samples" else 1):
            raise ValueError(f"aero.force_history.{key} must be an integer >= {2 if key == 'recent_samples' else 1}")
        result[key] = int(value)
    if result["min_samples"] < result["recent_samples"]:
        raise ValueError("aero.force_history.min_samples cannot be less than recent_samples")
    for key in ("relative_tolerance", "absolute_drag_tolerance_N_m"):
        value = result[key]
        if isinstance(value, bool):
            raise ValueError(f"aero.force_history.{key} must be finite and positive")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"aero.force_history.{key} must be finite and positive") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"aero.force_history.{key} must be finite and positive")
        result[key] = value
    return result


def _validated_rows(rows):
    normalized = []
    for row in rows:
        try:
            values = {key: float(row[key]) for key in ("iteration", *FORCE_COLUMNS)}
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("Force history is missing valid iteration/force values") from exc
        iteration = values["iteration"]
        if (not all(math.isfinite(value) for value in values.values()) or iteration < 0
                or not iteration.is_integer()
                or (normalized and iteration <= normalized[-1]["iteration"])):
            raise ValueError("Force-history iterations must be finite, actual and strictly increasing")
        values["iteration"] = int(iteration)
        normalized.append(values)
    return normalized


def assess_force_stability(rows, requested):
    """Screen final drag and recent oscillations; this does not prove convergence.

    A near-zero final drag uses an absolute force tolerance because division by
    zero, or replacing zero with an arbitrary normalizing force, is misleading.
    """
    configured = settings({"aero": {"force_history": requested}})
    if not configured["enabled"]:
        return dict(status="disabled", reason="Force-history recording is disabled")
    rows = _validated_rows(rows)
    result = dict(status="insufficient_history", sample_count=len(rows),
                  required_samples=configured["min_samples"],
                  recent_samples=configured["recent_samples"],
                  relative_tolerance=configured["relative_tolerance"],
                  absolute_drag_tolerance_N_m=configured["absolute_drag_tolerance_N_m"],
                  interpretation="Drag stability versus CFD iteration; not residual convergence or mesh independence.")
    if len(rows) < configured["min_samples"]:
        result["reason"] = "Not enough measured force-history samples"
        return result
    recent = rows[-configured["recent_samples"]:]
    values = [row[FORCE_COLUMNS[0]] for row in recent]
    final, recent_mean = values[-1], mean(values)
    deviation, spread = abs(final - recent_mean), max(values) - min(values)
    absolute = configured["absolute_drag_tolerance_N_m"]
    near_zero = abs(final) <= absolute
    if near_zero:
        passed = deviation <= absolute and spread <= absolute
        relative_change = relative_spread = None
    else:
        relative_change, relative_spread = deviation / abs(final), spread / abs(final)
        passed = (relative_change <= configured["relative_tolerance"]
                  and relative_spread <= configured["relative_tolerance"])
    result.update(status="passed" if passed else "failed", final_iteration=rows[-1]["iteration"],
                  first_recent_iteration=recent[0]["iteration"],
                  final_drag_per_unit_span_N_m=final, recent_mean_drag_per_unit_span_N_m=recent_mean,
                  final_to_recent_mean_difference_N_m=deviation,
                  recent_peak_to_peak_drag_N_m=spread,
                  relative_final_to_recent_mean_change=relative_change,
                  relative_recent_peak_to_peak_change=relative_spread,
                  normalization="absolute force tolerance near zero drag" if near_zero else "absolute final drag",
                  reason="Recent drag is stable under both checks" if passed else
                         "Final-to-recent-mean or recent peak-to-peak drag exceeds tolerance")
    return result


def configure_history(solver, cfg, wall_names):
    """Register native dimensional drag/lift reports before the single solve call."""
    configured = settings(cfg)
    if (not configured["enabled"] or not cfg.get("aero", {}).get("enabled", True)
            or cfg.get("_calibration_kind") == "cold_wall"):
        return dict(status="disabled", settings=configured)
    from .aero import configure_force_reports
    names, state = configure_force_reports(solver, cfg, wall_names)
    collection = solver.settings.solution.monitor.report_files
    name = "nose-study-force-history"
    if name not in collection.get_object_names():
        collection.create(name=name)
    report = collection[name]
    report.active = False
    raw_path = Path(cfg["project"]["work_dir"]) / RAW_NAME
    raw_path.unlink(missing_ok=True)
    report.file_name = str(raw_path.resolve())
    report.report_defs = [names["drag"], names["lift"]]
    allowed = report.frequency_of.allowed_values()
    frequency = next((value for value in allowed if str(value).casefold() in ("iteration", "iterations")), None)
    if frequency is None:
        raise ValueError(f"Fluent force report exposes no iteration frequency: {allowed}")
    report.frequency_of = frequency
    # Native frequency one captures early stopping and the actual last iterate.
    # Decimation happens only after the native file is safely retained.
    report.frequency = 1
    report.print = False
    report.active = True
    return dict(status="configured", settings=configured, raw_path=raw_path, report=report,
                names=names, reference_state=state, body_wall_zones=list(wall_names))


def parse_report_file(path, names, report_depth_m=1.0):
    """Parse named native force columns; metadata and indices are never forces."""
    depth = float(report_depth_m)
    if not math.isfinite(depth) or depth <= 0:
        raise ValueError("Force-history reference depth must be finite and positive")
    header, rows = None, []
    columns = {FORCE_COLUMNS[0]: names["drag"], FORCE_COLUMNS[1]: names["lift"]}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        tokens = shlex.split(line.strip().strip("()"), comments=True)
        if not tokens:
            continue
        labels = {}
        for column, token in enumerate(tokens):
            if token.casefold() in ("iteration", "iterations"):
                labels["iteration"] = column
            for key, name in columns.items():
                if token == name or token.startswith(name + " "):
                    labels[key] = column
        if len(labels) == 3:
            header = labels
            continue
        if header is None:
            continue
        try:
            float(tokens[0])
        except ValueError:
            continue
        if len(tokens) <= max(header.values()):
            raise ValueError("Incomplete Fluent force-history row")
        values = {key: float(tokens[column]) for key, column in header.items()}
        for key in FORCE_COLUMNS:
            values[key] /= depth
        rows.append(values)
    rows = _validated_rows(rows)
    if not rows:
        raise ValueError("No measured force-history records found in Fluent report file")
    return rows


def _sample_rows(rows, interval):
    """Select actual interval iterations and preserve the measured last sample."""
    sampled = [row for row in rows if row["iteration"] % interval == 0]
    if not sampled or sampled[-1]["iteration"] != rows[-1]["iteration"]:
        sampled.append(rows[-1])
    return sampled


def finish_history(solver, cfg, context):
    """Flush, independently verify the final iteration, and save sampled forces.

    Adapter/export failures leave a recoverable native file and unverified
    stability; they do not discard a completed CFD solution.
    """
    if context.get("status") == "disabled":
        return dict(status="disabled", force_stability=dict(status="disabled"))
    configured = context["settings"]
    raw_path = Path(context["raw_path"])
    metadata = dict(status="unavailable", settings=configured,
                    raw_report_file=raw_path.name,
                    conditions=temperature_conditions(cfg),
                    body_wall_zones=context["body_wall_zones"],
                    report_depth_m=context["reference_state"]["report_depth_m"],
                    native_frequency_iterations=1,
                    force_units="N/m of planar span",
                    method="Fluent native dimensional-force report file",
                    iteration_counter="Fluent native report-file Iteration column",
                    interpretation="Steady CFD convergence versus actual Fluent iteration; not physical time.")
    try:
        if context.get("report") is not None:
            context["report"].active = False
        from .aero import report_scalar
        compute = solver.settings.solution.report_definitions.compute
        final = {}
        for key, kind in zip(FORCE_COLUMNS, ("drag", "lift")):
            name = context["names"][kind]
            raw = compute(report_defs=[name])
            final[key] = report_scalar(raw, name) / metadata["report_depth_m"]
        # Compute() guarantees report values, not an iteration-counter schema.
        # Integer list elements may be other native report metadata. Preserve
        # the final dimensional values even if parsing a buffered file fails.
        metadata["final_report_forces"] = final
        rows = parse_report_file(raw_path, context["names"], metadata["report_depth_m"])
        metadata["final_force_comparison"] = {key: {
            "last_native_file_N_m": rows[-1][key], "final_report_N_m": final[key],
            "absolute_difference_N_m": abs(final[key] - rows[-1][key])}
            for key in FORCE_COLUMNS}
        matches_last = all(math.isclose(final[key], rows[-1][key], rel_tol=1e-6, abs_tol=1e-8)
                           for key in FORCE_COLUMNS)
        # The file records every iteration. Its actual index and force values
        # are authoritative; never append a sample using Compute() metadata or
        # an iteration budget. Both final forces must independently agree.
        verified = matches_last
        verification_method = ("Frequency-one native history final forces matched to final report computation"
                               if verified else "Final native force-history sample differs from final force reports")
        measured_final = rows[-1]["iteration"] if verified else None
        sampled = _sample_rows(rows, configured["sample_every_iterations"])
        csv_path = raw_path.with_name(CSV_NAME)
        atomic_csv(csv_path, sampled)
        stability = assess_force_stability(sampled, configured) if verified else dict(
            status="not_verified", reason=verification_method)
        metadata.update(status="complete" if verified else "partial", csv=csv_path.name,
                        csv_sha256=file_sha256(csv_path), raw_report_sha256=file_sha256(raw_path),
                        sample_count=len(sampled), native_sample_count=len(rows),
                        first_iteration=rows[0]["iteration"], last_recorded_iteration=rows[-1]["iteration"],
                        final_iteration=measured_final,
                        final_iteration_verified=verified, final_state_verification_method=verification_method,
                        final_force_matching_tolerance=dict(relative=1e-6, absolute_N_m=1e-8),
                        force_stability=stability)
    except (OSError, ValueError, RuntimeError, AttributeError, KeyError, TypeError) as exc:
        metadata.update(error=str(exc), force_stability=dict(status="not_verified", reason=str(exc)))
        if raw_path.is_file():
            metadata["raw_report_sha256"] = file_sha256(raw_path)
    return metadata


def read_saved_force_history(cfg):
    """Return provenance-checked measured rows, never reconstructed legacy values."""
    try:
        work = Path(cfg["project"]["work_dir"])
        summary = json.loads((work / "summary.json").read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("Saved solver summary must be a mapping")
        metadata = summary.get("force_history", {})
        if not isinstance(metadata, dict):
            raise ValueError("Saved force-history metadata must be a mapping")
        if metadata.get("status") != "complete" or not metadata.get("final_iteration_verified"):
            detail = metadata.get("error") or metadata.get("force_stability", {}).get("reason")
            status = metadata.get("status", "missing")
            raise ValueError(f"Force-history recording is {status}: " +
                (str(detail) if detail else "No complete final-state-verified force history was recorded"))
        base = cfg.get("geometry", {}).get("body_wall_name", "body_wall")
        zones = metadata.get("body_wall_zones")
        if (not isinstance(zones, list) or not zones or len(set(zones)) != len(zones)
                or any(not isinstance(zone, str) or (zone != base and not zone.startswith(base + "."))
                       for zone in zones)):
            raise ValueError("Force-history integration surfaces are not the matching body walls")
        if "body_wall_zones" in summary and set(summary["body_wall_zones"]) != set(zones):
            raise ValueError("Force-history walls differ from the saved solver body walls")
        if metadata.get("native_frequency_iterations") != 1:
            raise ValueError("Force history was not recorded every native iteration")
        if metadata.get("source_sha256") != {
                "final_case": file_sha256(work / "04_final.cas.h5"),
                "final_data": file_sha256(work / "04_final.dat.h5")}:
            raise ValueError("Force history does not match the current saved CFD solution")
        if metadata.get("conditions") != temperature_conditions(cfg):
            raise ValueError("Force-history conditions differ from the current case configuration")
        if metadata.get("csv") != CSV_NAME or metadata.get("raw_report_file") != RAW_NAME:
            raise ValueError("Force-history files have unexpected names")
        csv_path, raw_path = work / CSV_NAME, work / RAW_NAME
        if (metadata.get("csv_sha256") != file_sha256(csv_path)
                or metadata.get("raw_report_sha256") != file_sha256(raw_path)):
            raise ValueError("Saved force history changed after its provenance was recorded")
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = _validated_rows(list(csv.DictReader(handle)))
        if not rows or rows[-1]["iteration"] != metadata.get("final_iteration"):
            raise ValueError("Force-history CSV omits the independently measured final iteration")
        if len(rows) != metadata.get("sample_count"):
            raise ValueError("Force-history sample count differs from its recorded metadata")
        if float(metadata.get("report_depth_m", 0)) != 1.0:
            raise ValueError("Force history does not use the one-metre planar reference depth")
        return rows, metadata
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError(f"Saved force history is unavailable: {exc}") from exc


def assess_saved_force_history(cfg):
    """Reapply current tolerances to matching history; missing history cannot pass."""
    configured = settings(cfg)
    if not configured["enabled"] or not cfg.get("aero", {}).get("enabled", True):
        return dict(status="disabled", reason="Force-history recording is disabled")
    try:
        rows, metadata = read_saved_force_history(cfg)
        recorded = metadata.get("settings", {})
        if recorded.get("sample_every_iterations") != configured["sample_every_iterations"]:
            raise ValueError("Force-history sampling interval differs from the requested interval")
        result = assess_force_stability(rows, configured)
        result["history_csv"] = CSV_NAME
        result["history_csv_sha256"] = metadata["csv_sha256"]
        return result
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return dict(status="not_verified", reason=str(exc))


def force_history_diagnostics(cfg):
    """Read the exact recording/provenance failure without starting Ansys."""
    work = Path(cfg["project"]["work_dir"])
    try:
        summary = json.loads((work / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summary = {}
    metadata = summary.get("force_history", {})
    result = dict(case=cfg["project"].get("name"), work_dir=str(work),
                  force_stability=assess_saved_force_history(cfg),
                  recording_status=metadata.get("status", "missing"),
                  recording_error=metadata.get("error") or metadata.get("force_stability", {}).get("reason"),
                  final_state_verification_method=metadata.get("final_state_verification_method"),
                  sample_count=metadata.get("sample_count"),
                  final_iteration=metadata.get("final_iteration"), files={})
    result["final_force_comparison"] = metadata.get("final_force_comparison")
    for name, field in ((CSV_NAME, "csv_sha256"), (RAW_NAME, "raw_report_sha256")):
        path = work / name
        info = dict(path=str(path), exists=path.is_file())
        if path.is_file():
            info["recorded_hash_matches"] = metadata.get(field) == file_sha256(path)
            if name == RAW_NAME:
                lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
                info.update(first_lines=lines[:12], last_lines=lines[-6:])
        result["files"][name] = info
    return result


def recheck_saved_force_history(solver, cfg, wall_names):
    """Reverify an existing recording against its saved checkpoint, without iterating.

    An earlier adapter failure can leave a valid raw file. Recovery requires
    the recorded raw hash, conditions and body walls, a matching CFD pair, and
    independent final forces from the already loaded, validated native case.
    Source mismatches cannot be repaired by relabeling old history.
    """
    current = assess_saved_force_history(cfg)
    if current["status"] != "not_verified":
        return current
    work = Path(cfg["project"]["work_dir"])
    summary_path = work / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        previous = summary.get("force_history", {})
        raw_path = work / RAW_NAME
        source = {"final_case": file_sha256(work / "04_final.cas.h5"),
                  "final_data": file_sha256(work / "04_final.dat.h5")}
        binding = previous.get("source_sha256") or summary.get("numerical_quality", {}).get("source_sha256")
        if (binding != source or previous.get("conditions") != temperature_conditions(cfg)
                or previous.get("raw_report_file") != RAW_NAME
                or previous.get("raw_report_sha256") != file_sha256(raw_path)
                or previous.get("native_frequency_iterations") != 1
                or previous.get("report_depth_m") != 1.0
                or set(previous.get("body_wall_zones", [])) != set(wall_names)):
            return current
    except (OSError, ValueError, KeyError, TypeError):
        return current
    from .aero import configure_force_reports
    names, state = configure_force_reports(solver, cfg, wall_names)
    context = dict(settings=settings(cfg), raw_path=raw_path, report=None,
                   names=names, reference_state=state, body_wall_zones=list(wall_names))
    metadata = finish_history(solver, cfg, context)
    metadata["source_sha256"] = source
    metadata["rechecked_without_iterations"] = True
    metadata["previous_verification"] = {
        "status": previous.get("status"), "error": previous.get("error"),
        "force_stability": previous.get("force_stability")}
    # Retain the previous metadata before replacing only the derived history.
    from datetime import datetime, timezone
    from .utils import atomic_json
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_json(work / "solver_diagnostics" / ("force_history_before_recheck_" + stamp + ".json"), previous)
    summary["force_history"] = metadata
    atomic_json(summary_path, summary)
    return assess_saved_force_history(cfg)
