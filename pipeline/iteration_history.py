"""Record measured Fluent iterations through native report files."""
import math
from pathlib import Path
import shlex
from contextlib import contextmanager

from .temperature_export import atomic_csv
from .utils import file_sha256


REPORTS = {
    "wall_temperature_min_k": ("auto-wall-temp-min", "surface-facetmin"),
    "wall_temperature_avg_k": ("auto-wall-temp-avg", "surface-areaavg"),
    "wall_temperature_max_k": ("auto-wall-temp-max", "surface-facetmax"),
}


def configure_history(solver, cfg, wall_names):
    definitions = solver.settings.solution.report_definitions.surface
    for name, report_type in REPORTS.values():
        definitions[name] = dict(surface_names=wall_names, report_type=report_type, field="temperature")
    collection = solver.settings.solution.monitor.report_files
    name = "nose-study-temperature-history"
    if name not in collection.get_object_names():collection.create(name=name)
    report = collection[name]
    report.active = False
    raw_path = Path(cfg["project"]["work_dir"]) / "solver_temperature_history.out"
    raw_path.unlink(missing_ok=True)
    report.file_name = str(raw_path.resolve())
    report.report_defs = [name for name, _ in REPORTS.values()]
    allowed = report.frequency_of.allowed_values()
    frequency = next((value for value in allowed if str(value).casefold() in ("iteration", "iterations")), None)
    if frequency is None:raise ValueError(f"No iteration report-file frequency is available: {allowed}")
    report.frequency_of = frequency
    report.frequency = int(cfg["solver"].get("temperature_history", {}).get("sample_every_iterations", 25))
    report.print = False
    report.active = True
    return raw_path, report


def parse_report_file(path):
    """Use actual column labels and iteration numbers, never an iteration budget."""
    header, rows = None, []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        tokens = shlex.split(line.strip().strip("()"), comments=True)
        if not tokens:continue
        labels = {}
        for column, token in enumerate(tokens):
            if token.casefold() in ("iteration", "iterations"):
                labels["iteration"] = column
            for key, (name, _) in REPORTS.items():
                if token == name or token.startswith(name + " "):
                    labels[key] = column
        if len(labels) == 4:
            header = labels
            continue
        if header is None:continue
        try:
            # Native report files may contain a title/footer after the header.
            float(tokens[0])
        except ValueError:
            continue
        if len(tokens) <= max(header.values()):raise ValueError("Incomplete Fluent history row")
        values = {key:float(tokens[column]) for key,column in header.items()}
        iteration = values["iteration"]
        if (not all(math.isfinite(value) for value in values.values()) or iteration < 0
                or not iteration.is_integer() or (rows and iteration <= rows[-1]["iteration"])):
            raise ValueError("Invalid or non-increasing measured iterations")
        if not (0 < values["wall_temperature_min_k"] <= values["wall_temperature_avg_k"] <= values["wall_temperature_max_k"]):
            raise ValueError("Nonphysical temperature history sample")
        values["iteration"] = int(iteration)
        rows.append(values)
    if not rows:raise ValueError("No measured temperature history records found in Fluent report file")
    return rows


def _finish_temperature_history(raw_path, report):
    report.active = False
    rows = parse_report_file(raw_path)
    path = raw_path.with_suffix(".csv")
    atomic_csv(path, rows)
    return dict(status="complete", csv=path.name, csv_sha256=file_sha256(path),
                raw_report_file=raw_path.name, sample_count=len(rows),
                method="Fluent native report file", iteration_counter="Fluent report-file Iteration column",
                interpretation="Steady CFD convergence versus measured Fluent iteration; not physical time.")


@contextmanager
def _defer_residual_stop(solver, enabled):
    """Collect a requested force window without an immediate residual-only stop.

    Equation monitoring and residual thresholds remain unchanged. Restore the
    original stop flags even on an iterate failure, before saving a checkpoint.
    """
    original = []
    try:
        if enabled:
            equations = solver.settings.solution.monitor.residual.equations
            for name in equations.get_object_names():
                equation = equations[name]
                value = equation.check_convergence.get_state()
                if not isinstance(value, bool):
                    raise ValueError(f"Unexpected residual convergence-stop flag for {name}: {value!r}")
                original.append((name, equation, value))
                equation.check_convergence = False
                if equation.check_convergence.get_state() is not False:
                    raise ValueError(f"Could not defer residual-only stopping for force recording: {name}")
            if not original:
                raise ValueError("No residual stopping controls are available for the force-recording window")
            print("[force-history] Recording the requested continuation window; residual thresholds remain unchanged", flush=True)
        yield {"residual_stopping_temporarily_disabled": bool(enabled),
               "original_residual_stop_flags": {name: value for name, _, value in original}}
    finally:
        for name, equation, value in original:
            equation.check_convergence = value
            if equation.check_convergence.get_state() is not value:
                raise ValueError(f"Could not restore the residual convergence-stop flag for {name}")


def iterate_with_histories(solver, cfg, wall_names, *, record_forces=True, force_recording_window=False):
    """Configure both monitors around one unchanged native steady iterate call."""
    iterations = int(cfg["solver"].get("max_iterations", 1200))
    iterate = solver.settings.solution.run_calculation.iterate
    result = {"temperature_history": {"status": "disabled"}, "force_history": {"status": "disabled"}}
    temperature_context, force_context = None, None
    if record_forces:
        from .force_history import configure_history as configure_forces
        try:
            force_context = configure_forces(solver, cfg, wall_names)
        except Exception as exc:
            result["force_history"] = {"status": "unavailable", "force_stability": {"status": "not_verified", "reason": str(exc)}}
            print(f"[force-history] Native recording unavailable: {exc}; preserving the steady solve.", flush=True)
    if cfg["solver"].get("temperature_history", {}).get("enabled", True):
        try:
            temperature_context = configure_history(solver, cfg, wall_names)
            print(f"[temperature-history] Recording measured iterations: {temperature_context[0]}", flush=True)
        except Exception as exc:
            result["temperature_history"] = {"status": "unavailable", "error": str(exc)}
            print(f"[temperature-history] Native recording unavailable: {exc}; continuing the steady solve.", flush=True)
    # Keep Fluent's original single solve call and its convergence behavior.
    # Iteration failures are deliberately not caught as optional export errors.
    collect_window = (force_recording_window and force_context is not None
        and force_context.get("status") == "configured"
        and cfg.get("aero", {}).get("force_history", {}).get("require_stable", False))
    with _defer_residual_stop(solver, collect_window) as recording_window:
        iterate(iter_count=iterations)
    if temperature_context is not None:
        try:
            result["temperature_history"] = _finish_temperature_history(*temperature_context)
        except Exception as exc:
            print(f"[temperature-history] Could not convert native history: {exc}; preserving the CFD solution.", flush=True)
            result["temperature_history"] = {"status": "unavailable", "raw_report_file": temperature_context[0].name, "error": str(exc)}
    if force_context is not None:
        from .force_history import finish_history
        result["force_history"] = finish_history(solver, cfg, force_context)
        if collect_window:
            result["force_history"]["recording_window"] = {
                **recording_window, "requested_iteration_budget": iterations,
                "residual_thresholds_unchanged": True}
    return result


def iterate_with_history(solver, cfg, wall_names):
    """Compatibility entry point for temperature-only history consumers."""
    return iterate_with_histories(solver, cfg, wall_names, record_forces=False)["temperature_history"]
