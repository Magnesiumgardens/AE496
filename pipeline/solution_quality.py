"""Distinguish completed iterations from a numerically usable CFD solution."""
import math
from pathlib import Path

from .reporting import read_json


def residual_records(path):
    path = Path(path)
    if not path.is_file():return [], None
    header, rows, last_limit = None, [], None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = line.split()
        if tokens and tokens[0].casefold() == "iter" and "continuity" in tokens and "energy" in tokens:
            header = tokens[:tokens.index("time/iter")] if "time/iter" in tokens else tokens
            continue
        lower = line.casefold()
        if "temperature limited" in lower or "absolute pressure limited" in lower:
            last_limit = rows[-1]["iteration"] if rows else 0
        if header is None or len(tokens) < len(header):continue
        try:
            number = float(tokens[0])
            values = [float(value) for value in tokens[1:len(header)]]
        except ValueError:continue
        if not number.is_integer() or number < 0:continue
        if rows and int(number) < rows[-1]["iteration"]:
            rows, last_limit = [], None  # a new solver run in a combined transcript
        if rows and int(number) == rows[-1]["iteration"]:rows.pop()
        rows.append(dict(iteration=int(number), **dict(zip(header[1:],values))))
    return rows, last_limit


def assess_log(path, cfg):
    rows, last_limit = residual_records(path)
    if not rows:return {"status":"not_verified", "reason":"No residual table found"}
    last = rows[-1]
    thresholds = cfg.get("solver", {}).get("residual_thresholds", {})
    failures = [f"{name}={value:g}" for name,value in last.items() if name != "iteration"
                and (not math.isfinite(value) or value < 0 or value > float(thresholds.get(name, 1e-6 if name == "energy" else 1e-3)))]
    recent_limits = last_limit is not None and last["iteration"] - last_limit <= 25
    reasons = []
    if failures:reasons.append("Final residuals exceed screening thresholds: " + ", ".join(failures))
    if recent_limits:reasons.append("Temperature/pressure limiting persists near the final iteration")
    trend = {}
    if len(rows) >= 60:
        from statistics import median
        window = min(300,len(rows)//3)
        for key in last:
            if key == "iteration":continue
            previous = [r[key] for r in rows[-2*window:-window] if math.isfinite(r[key]) and r[key] > 0]
            current = [r[key] for r in rows[-window:] if math.isfinite(r[key]) and r[key] > 0]
            if previous and current:
                before,after = median(previous),median(current)
                trend[key] = dict(previous_window_median=before,last_window_median=after,
                                  last_to_previous_ratio=after/before,window_samples=window)
    invalid_residuals = any(not math.isfinite(value) or value < 0 for key,value in last.items() if key != "iteration")
    return dict(numerical_state="unstable" if recent_limits or invalid_residuals else "not_converged" if reasons else "residual_screen_passed",
        residual_trend=trend,status="failed" if reasons else "residual_screen_passed", final_iteration=last["iteration"],
        final_residuals={key:(value if math.isfinite(value) else None) for key,value in last.items() if key != "iteration"},
        recent_temperature_or_pressure_limiting=recent_limits, reasons=reasons,
        interpretation="Residual screening is not force/temperature stability or mesh-independence verification.")


def saved_quality(cfg):
    work = Path(cfg["project"]["work_dir"])
    log = work / "batch_logs/solver.log"
    summary = read_json(work / "summary.json")
    recorded = summary.get("numerical_quality")
    if isinstance(recorded, dict):
        from .utils import file_sha256
        try:
            if recorded.get("source_sha256") == {
                "final_case":file_sha256(work/"04_final.cas.h5"),
                "final_data":file_sha256(work/"04_final.dat.h5")}:
                return recorded
        except OSError:pass
    try:
        # An older transcript must not assess a newly replaced case/data pair.
        if log.stat().st_mtime_ns < max((work/name).stat().st_mtime_ns for name in ("04_final.cas.h5","04_final.dat.h5")):
            return {"status":"not_verified", "reason":"Solver transcript predates saved solution"}
    except OSError:return {"status":"not_verified", "reason":"No matching solver transcript"}
    return assess_log(log,cfg)


def require_usable_solution(cfg):
    quality = saved_quality(cfg)
    if quality["status"] == "failed":
        raise ValueError("Saved CFD solution is unconverged: " + "; ".join(quality.get("reasons",[])))
    return quality


def save_diagnostics(log_path,cfg):
    """Plot actual recorded residuals, including runs rejected for downstream use."""
    rows,_ = residual_records(log_path)
    if not rows:return {}
    from .temperature_export import atomic_csv
    from .utils import atomic_json
    work = Path(cfg["project"]["work_dir"])
    directory = work/"solver_diagnostics"
    directory.mkdir(parents=True,exist_ok=True)
    atomic_csv(directory/"residual_history.csv",rows)
    quality = assess_log(log_path,cfg)
    atomic_json(directory/"residual_summary.json",quality)
    try:
        from .reporting import _plotting
        plt = _plotting()
        fig,ax = plt.subplots(figsize=(10,5))
        for name in rows[-1]:
            if name != "iteration":
                values=[r[name] if math.isfinite(r[name]) and r[name] > 0 else float("nan") for r in rows]
                ax.plot([r["iteration"] for r in rows],values,label=name)
        ax.set_yscale("log",nonpositive="mask")
        ax.set(xlabel="Measured Fluent iteration",ylabel="Reported residual",
               title=f"{cfg['project']['name']} — CFD residuals ({quality['numerical_state'].replace('_',' ')})")
        ax.grid(alpha=.25,which="both");ax.legend();fig.tight_layout()
        fig.savefig(directory/"residual_history.png",dpi=160);plt.close(fig)
    except (OSError, ValueError, ImportError) as exc:
        print(f"[diagnostics] Residual CSV saved; optional plot unavailable: {exc}",flush=True)
    return quality
