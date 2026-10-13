"""Record measured Fluent iterations through native report files."""
import math
from pathlib import Path
import shlex

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


def iterate_with_history(solver, cfg, wall_names):
    iterations = int(cfg["solver"].get("max_iterations", 1200))
    iterate = solver.settings.solution.run_calculation.iterate
    if not cfg["solver"].get("temperature_history", {}).get("enabled", True):
        iterate(iter_count=iterations)
        return {"status":"disabled"}
    try:
        raw_path, report = configure_history(solver, cfg, wall_names)
    except Exception as exc:
        print(f"[temperature-history] Native recording unavailable: {exc}; continuing the steady solve.", flush=True)
        iterate(iter_count=iterations)
        return {"status":"unavailable", "error":str(exc)}
    print(f"[temperature-history] Recording measured iterations: {raw_path}", flush=True)
    # Keep Fluent's original single solve call and its convergence behavior.
    # Iteration failures are deliberately not caught as optional export errors.
    iterate(iter_count=iterations)
    try:
        report.active = False
        rows = parse_report_file(raw_path)
        path = raw_path.with_suffix(".csv")
        atomic_csv(path, rows)
        return dict(status="complete", csv=path.name, csv_sha256=file_sha256(path),
                    raw_report_file=raw_path.name, sample_count=len(rows),
                    method="Fluent native report file", iteration_counter="Fluent report-file Iteration column",
                    interpretation="Steady CFD convergence versus measured Fluent iteration; not physical time.")
    except Exception as exc:
        print(f"[temperature-history] Could not convert native history: {exc}; preserving the CFD solution.", flush=True)
        return {"status":"unavailable", "raw_report_file":raw_path.name, "error":str(exc)}
