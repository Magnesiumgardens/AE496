"""Distinguish completed iterations from a numerically usable CFD solution."""
import math
from pathlib import Path

from .reporting import read_json


def assess_log(path, cfg):
    path = Path(path)
    if not path.is_file():return {"status":"not_verified", "reason":"No solver transcript available"}
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
        rows.append(dict(iteration=int(number), **dict(zip(header[1:],values))))
    if not rows:return {"status":"not_verified", "reason":"No residual table found"}
    last = rows[-1]
    thresholds = cfg.get("solver", {}).get("residual_thresholds", {})
    failures = [f"{name}={value:g}" for name,value in last.items() if name != "iteration"
                and (not math.isfinite(value) or value > float(thresholds.get(name, 1e-6 if name == "energy" else 1e-3)))]
    recent_limits = last_limit is not None and last["iteration"] - last_limit <= 25
    reasons = []
    if failures:reasons.append("Final residuals exceed screening thresholds: " + ", ".join(failures))
    if recent_limits:reasons.append("Temperature/pressure limiting persists near the final iteration")
    return dict(status="failed" if reasons else "residual_screen_passed", final_iteration=last["iteration"],
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
