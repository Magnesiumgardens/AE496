"""Sequential case orchestration with stage-specific, content-addressed resume."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import yaml

from .reporting import (build_result_row, create_comparison_outputs, load_yaml,
                        read_json, save_batch_summary)

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("mesh", "solver", "thermal_export", "aero", "thermal")
INVALIDATES = {
    "mesh": ("mesh", "solver", "thermal_export", "aero", "thermal"),
    "solver": ("solver", "thermal_export", "aero", "thermal"),
    "thermal_export": ("thermal_export", "thermal"),
    "aero": ("aero",), "thermal": ("thermal",),
}


class BatchError(RuntimeError):
    pass


def resolve(root, value):
    path = Path(value)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def safe_case_name(stem):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    if not name:
        raise BatchError(f"Cannot make a case name from {stem!r}")
    return name


def validate_thermal_inputs(cfg):
    mode = cfg.get("loading", {}).get("mode", "fixed_heat_flux")
    if mode != "fixed_heat_flux":
        raise BatchError(f"This workflow supports fixed_heat_flux, not {mode!r}. Historical endpoint data remain report-readable.")
    for section, names in (("material", ("conductivity_w_mk", "density_kg_m3", "specific_heat_j_kgk")),
                           ("transient", ("duration_s",))):
        for name in names:
            value = cfg.get(section, {}).get(name)
            try:
                valid = float(value) > 0 and float(value) < float("inf")
            except (ValueError, TypeError):
                valid = False
            if not valid:
                raise BatchError(f"A finite positive {section}.{name} is required")


def file_hash(path):
    path = Path(path)
    if not path.is_file():
        raise BatchError(f"Missing required input: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_files(work, cfg):
    export = cfg.get("thermal_export", {}).get("output_directory", "thermal_bc")
    export_dir = resolve(work, export)
    return {
        "mesh": [work / "02_mesh.msh.h5"],
        "solver": [work / "04_final.cas.h5", work / "04_final.dat.h5"],
        "thermal_export": [export_dir / "body_wall_faces.csv"],
        "aero": [work / "aero_summary.json"],
        "thermal": [work / "thermal_analysis/thermal_summary.json", work / "thermal_analysis/temperature_history.csv",
                    work / "thermal_analysis/wall_temperature_history.csv"],
    }


def stage_fingerprint(stage, cfg, thermal_cfg, outputs):
    """Exclude unrelated settings; hash all actual upstream file contents."""
    keys = {
        "mesh": ("geometry", "mesh"),
        "solver": ("flight", "physics", "thermal", "solver", "geometry"),
        "thermal_export": ("geometry", "thermal_export", "flight", "physics", "thermal"),
        "aero": ("geometry", "flight", "physics", "thermal", "aero"),
        "thermal": ("thermal", "geometry", "reporting"),
    }[stage]
    values = {key: copy.deepcopy(cfg.get(key, {})) for key in keys}
    if "geometry" in values:
        values["geometry"].pop("input_file", None)
    inputs = {"geometry": file_hash(cfg["geometry"]["input_file"])}
    dependencies = {"mesh": (), "solver": ("mesh",), "thermal_export": ("solver",),
                    "aero": ("solver",), "thermal": ("thermal_export",)}[stage]
    for dependency in dependencies:
        inputs[dependency] = {str(path.name): file_hash(path) for path in outputs[dependency]}
    if stage == "thermal":
        values["thermal_config"] = thermal_cfg
    return hashlib.sha256(json.dumps(dict(version=2, settings=values, inputs=inputs),
                                       sort_keys=True, allow_nan=False).encode()).hexdigest()


def output_hashes(paths):
    return {str(path): file_hash(path) for path in paths}


def validate_stage_outputs(stage, paths):
    """A nonempty file is insufficient evidence of a usable exported load."""
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        raise BatchError(f"{stage} required output files are missing or empty")
    if stage == "thermal_export":
        from .thermal_loading import read_boundary
        try:
            read_boundary(paths[0])
        except (OSError, ValueError, RuntimeError) as exc:
            raise BatchError(f"Invalid thermal_export boundary CSV: {exc}") from exc
    elif stage in ("aero", "thermal"):
        summary = read_json(paths[0])
        if summary.get("status") != "complete":
            raise BatchError(f"{stage} summary does not report a complete result")
        if stage == "thermal" and summary.get("loading_mode") != "fixed_heat_flux":
            raise BatchError("Thermal result does not use the requested direct fixed_heat_flux mode")


def reusable(record, fingerprint, paths):
    if not record or record.get("fingerprint") != fingerprint or record.get("status") != "complete":
        return False
    try:
        return record.get("output_sha256") == output_hashes(paths)
    except BatchError:
        return False


def _save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def _terminate_process_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        import signal
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            import signal
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.kill()
        process.wait(timeout=5)


def run_stage(command, log_path, hard_timeout_s=7200, inactivity_timeout_s=1200):
    """Capture output and stop only this owned subprocess tree on timeout."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1,
                               start_new_session=os.name != "nt")
    messages = queue.Queue()

    def collect():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=collect, daemon=True).start()
    started = last_output = time.monotonic()
    ended = False
    try:
        with log_path.open("w", encoding="utf-8") as log:
            while not ended or process.poll() is None:
                try:
                    line = messages.get(timeout=0.2)
                    if line is None:
                        ended = True
                    else:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                        last_output = time.monotonic()
                except queue.Empty:
                    pass
                now = time.monotonic()
                if now - started > hard_timeout_s or now - last_output > inactivity_timeout_s:
                    reason = "hard" if now - started > hard_timeout_s else "inactivity"
                    log.write(f"\nStage {reason} timeout\n")
                    raise BatchError(f"Stage {reason} timeout; see {log_path}")
            code = process.wait()
            if code:
                raise BatchError(f"Stage exited with status {code}; see {log_path}")
    finally:
        _terminate_process_tree(process)
        if process.stdout:
            process.stdout.close()


def _failed_during_meshing(case_dir, stage_name="01_cfd"):
    """Legacy diagnostic only; no automatic physics-changing retry is performed."""
    path = Path(case_dir) / "batch_logs" / f"{stage_name}.log"
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    if any(marker in text for marker in ("attributeerror", "launchfluenterror", "license", "filenotfounderror",
                                        "launch process has timed out", "geometry service cannot be initialized")):
        return False
    return "[solve] launching" not in text and any(marker in text for marker in (
        "[mesh] adding boundary layers", "[mesh] generating boundary layers", "[mesh] boundary-layer"))


def _archive_stage(stage, work, cfg, state):
    """Retain stale stage and downstream artifacts before rerunning them."""
    paths = stage_files(work, cfg)
    candidates = []
    for item in INVALIDATES[stage]:
        candidates.extend(paths[item])
        candidates.append(work / "batch_logs" / f"{item}.log")
        state["stages"].pop(item, None)
    if stage == "mesh":
        candidates.extend([work / "01_labeled_geometry.pmdb", work / "geometry_summary.json"])
    if stage in ("mesh", "solver"):
        candidates.extend([work / "03_setup.cas.h5", work / "summary.json", work / "body_wall_heat_flux.csv"])
    if "thermal_export" in INVALIDATES[stage]:
        candidates.append(resolve(work, cfg.get("thermal_export", {}).get("output_directory", "thermal_bc")))
    if "thermal" in INVALIDATES[stage]:
        candidates.extend([work / "thermal_analysis", work / "reports"])
    existing = [path for path in dict.fromkeys(candidates) if path.exists()]
    if not existing:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = work / "_history" / f"{stamp}_{stage}"
    archive.mkdir(parents=True)
    state_path = work / "_batch/stage_state.json"
    if state_path.is_file():
        shutil.copy2(state_path, archive / "previous_stage_state.json")
    parents = [path for path in existing if not any(parent != path and parent in path.parents for parent in existing)]
    for path in parents:
        try:
            relative = path.relative_to(work)
        except ValueError as exc:
            raise BatchError(f"Refusing to archive output outside case directory: {path}") from exc
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    print(f"[resume] Retained previous outputs: {archive}")


def run_case(config_path, thermal_config_path="thermal_analysis.yaml", resume=True, stop_after="thermal",
             *, _runner=None, _enabled_stages=None, _timeouts=None, _allow_adopt=True):
    from .config import load_config
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    work = Path(cfg["project"]["work_dir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if stop_after not in STAGES:
        raise BatchError(f"Unknown stop-after stage: {stop_after}")
    thermal_path = resolve(REPO_ROOT, thermal_config_path)
    thermal_cfg = load_yaml(thermal_path) if stop_after == "thermal" else {}
    enabled = set(STAGES if _enabled_stages is None else _enabled_stages)
    if stop_after == "thermal" and "thermal" in enabled:
        validate_thermal_inputs(thermal_cfg)
    files = stage_files(work, cfg)
    state_path = work / "_batch/stage_state.json"
    state = read_json(state_path) or {"version": 2, "stages": {}}
    state.setdefault("stages", {})
    arguments = ["--config", str(config_path)]
    commands = {
        "mesh": ["run_all.py", *arguments, "--stop-after", "mesh"],
        "solver": ["run_solver_only.py", *arguments],
        "thermal_export": ["export_thermal_only.py", *arguments],
        "aero": ["export_aero_only.py", *arguments],
        "thermal": ["run_thermal_analysis.py", *arguments, "--thermal-config", str(thermal_path)],
    }
    started, outcomes = time.monotonic(), {}
    for stage in STAGES[:STAGES.index(stop_after) + 1]:
        fingerprint = stage_fingerprint(stage, cfg, thermal_cfg, files)
        record = state["stages"].get(stage)
        try:
            validate_stage_outputs(stage, files[stage])
            valid_outputs = True
        except BatchError:
            valid_outputs = False
        if resume and valid_outputs and reusable(record, fingerprint, files[stage]):
            print(f"[resume] {cfg['project']['name']}: {stage} unchanged")
            outcomes[stage] = "reused"
            continue
        can_adopt = resume and _allow_adopt and record is None and valid_outputs
        if can_adopt and stage == "mesh" and Path(cfg["geometry"]["input_file"]).stat().st_mtime_ns > files[stage][0].stat().st_mtime_ns:
            can_adopt = False
        if stage == "aero" and can_adopt:
            from .trade_study import aero_complete
            can_adopt = aero_complete(work, cfg)
        if stage == "thermal" and can_adopt:
            summary = read_json(files[stage][0])
            can_adopt = summary.get("status") == "complete" and summary.get("loading_mode") == "fixed_heat_flux"
        if can_adopt:
            print(f"[resume] {cfg['project']['name']}: adopting existing {stage}; original provenance unverified")
            state["stages"][stage] = dict(status="complete", fingerprint=fingerprint,
                output_sha256=output_hashes(files[stage]), provenance="adopted_unverified")
            _save_state(state_path, state)
            outcomes[stage] = "adopted_unverified"
            continue
        if stage not in enabled:
            raise BatchError(f"{stage} is disabled but matching outputs are unavailable for {work.name}")
        # Downstream files can exist even when this stage's output was removed.
        # They must not be adopted after regenerating an upstream prerequisite.
        _archive_stage(stage, work, cfg, state)
        _save_state(state_path, state)
        print(f"[study] {cfg['project']['name']}: {stage}", flush=True)
        timeout = (_timeouts or {}).get(stage, {})
        try:
            runner = _runner or run_stage
            if stage == "solver" and outcomes.get("mesh") == "executed":
                delay = min(60.0, max(0.0, float(cfg.get("runtime", {}).get("meshing_release_delay_s", 0))))
                if delay:
                    print(f"[study] Waiting {delay:g} s for meshing license release", flush=True)
                    time.sleep(delay)
            runner([sys.executable, "-u", str(REPO_ROOT / commands[stage][0]), *commands[stage][1:]],
                   work / "batch_logs" / f"{stage}.log",
                   hard_timeout_s=float(timeout.get("hard_s", 7200)),
                   inactivity_timeout_s=float(timeout.get("inactivity_s", 1200)))
            validate_stage_outputs(stage, files[stage])
        except Exception as exc:
            state["stages"][stage] = dict(status="failed", fingerprint=fingerprint, error=str(exc))
            _save_state(state_path, state)
            raise
        state["stages"][stage] = dict(status="complete", fingerprint=fingerprint,
                                     output_sha256=output_hashes(files[stage]), provenance="executed")
        _save_state(state_path, state)
        outcomes[stage] = "executed"
    return dict(case=cfg["project"]["name"], work_dir=str(work), status="complete", stages=outcomes,
                elapsed_s=round(time.monotonic() - started, 3))


def write_case_config(base_cfg, geometry_file, case_name, case_dir, config_path, force_fluent_gui=None):
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("project", {}).update(name=case_name, work_dir=str(Path(case_dir).resolve()))
    cfg.setdefault("geometry", {})["input_file"] = str(Path(geometry_file).resolve())
    if force_fluent_gui is not None:
        cfg.setdefault("runtime", {})["fluent_gui"] = bool(force_fluent_gui)
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    changed = config_path.is_file() and load_yaml(config_path) != cfg
    if changed:
        old = load_yaml(config_path)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        shutil.copy2(config_path, config_path.parent / f"config_previous_{stamp}.yaml")
        if not (config_path.parent / "stage_state.json").exists():
            changed_keys = {key for key in ("geometry", "mesh", "flight", "physics", "thermal", "solver",
                                            "aero", "thermal_export", "reporting") if old.get(key) != cfg.get(key)}
            affected = set()
            for key in changed_keys:
                first = ("mesh" if key in ("geometry", "mesh") else "solver" if key in ("flight", "physics", "thermal", "solver")
                         else "thermal_export" if key == "thermal_export" else "aero" if key == "aero" else "thermal")
                affected.update(INVALIDATES[first])
            if affected:
                _save_state(config_path.parent / "stage_state.json", {"version": 2, "stages": {
                    stage: {"status": "stale", "reason": "case configuration changed"} for stage in affected}})
    if not config_path.is_file() or changed:
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def main(batch_config_path="batch.yaml"):
    path = Path(batch_config_path).resolve()
    batch, root = load_yaml(path), path.parent
    base_path = resolve(root, batch.get("base_config", "config.yaml"))
    base = load_yaml(base_path)
    geometry_root = resolve(root, batch.get("geometry_directory", "geometry_batch"))
    geometries = sorted(geometry_root.glob(batch.get("geometry_glob", "*.x_t")))
    geometries = [item for item in geometries if item.is_file()]
    if not geometries:
        raise BatchError(f"No geometry inputs found under {geometry_root}")
    names = [safe_case_name(item.stem) for item in geometries]
    if len(names) != len(set(names)):
        raise BatchError("Geometry filenames produce colliding case names")
    runs = resolve(root, batch.get("runs_directory", "runs"))
    thermal = resolve(root, batch.get("thermal_config", "thermal_analysis.yaml"))
    stop_after = batch.get("stop_after", "thermal" if batch.get("run_thermal", True) else "aero")
    enabled = set(STAGES)
    if not batch.get("run_cfd", True):
        enabled -= {"mesh", "solver"}
    if not batch.get("run_thermal", True):
        enabled.discard("thermal")
    if batch.get("run_adiabatic_reference", False):
        print("[batch] Legacy adiabatic flag ignored: this study uses direct cold-wall heat flux.")
    if stop_after == "thermal":
        thermal_cfg = load_yaml(thermal)
        validate_thermal_inputs(thermal_cfg)
        if len(geometries) > 1 and thermal_cfg.get("boundary_csv"):
            raise BatchError("A global boundary_csv cannot be shared by multiple noses; use each case's thermal_bc/body_wall_faces.csv")
    rows = []
    for geometry, name in zip(geometries, names):
        directory, started = runs / name, time.monotonic()
        config_path = directory / "_batch/config.yaml"
        write_case_config(base, geometry, name, directory, config_path, batch.get("force_fluent_gui", False))
        try:
            result = run_case(config_path, thermal, batch.get("resume", True), stop_after,
                              _enabled_stages=enabled, _timeouts=batch.get("timeouts"))
            row = build_result_row(geometry, name, directory, result["status"], result["elapsed_s"])
        except Exception as exc:
            print(f"[batch] {name} failed: {exc}", flush=True)
            row = build_result_row(geometry, name, directory, "failed", time.monotonic() - started, str(exc))
        rows.append(row)
        save_batch_summary(runs / "comparison", rows)
        if row["status"] == "failed" and batch.get("stop_on_error", False):
            break
    if batch.get("comparison_outputs", True):
        create_comparison_outputs(runs, rows, runs / "comparison")
    return 1 if any(row["status"] == "failed" for row in rows) else 0
