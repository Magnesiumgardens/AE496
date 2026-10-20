"""Sequential case orchestration with stage-specific, content-addressed resume."""
from __future__ import annotations

import copy
import hashlib
import json
import math
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
STAGES = ("mesh", "solver", "temperature_export", "aero", "calibration", "thermal")
INVALIDATES = {
    "mesh": STAGES, "solver": STAGES[1:],
    "temperature_export": ("temperature_export", "calibration", "thermal"), "aero": ("aero",),
    "calibration": ("calibration", "thermal"), "thermal": ("thermal",),
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
    if cfg.get("loading", {}).get("mode") != "convection":
        raise BatchError("Mechanical requires loading.mode=convection and a specified h")


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
    from .temperature_export import temperature_paths
    from .aerothermal_calibration import calibration_paths
    calibration = calibration_paths(cfg)
    csv_path, summary_path = temperature_paths(cfg)
    return {
        "mesh": [work / "02_mesh.msh.h5"],
        "solver": [work / "04_final.cas.h5", work / "04_final.dat.h5"],
        "temperature_export": [csv_path, summary_path],
        "aero": [work / "aero_summary.json"],
        "calibration": [calibration[key] for key in ("faces", "nodes", "summary", "cold_summary")]
                       + [calibration["cold_work"] / name for name in ("04_final.cas.h5", "04_final.dat.h5")],
        "thermal": [work / "thermal_analysis/thermal_summary.json", work / "thermal_analysis/wall_temperature_history.csv", work / "thermal_analysis/thermal_nodes_final.csv"],
    }


def stage_fingerprint(stage, cfg, thermal_cfg, outputs):
    """Exclude unrelated settings; hash all actual upstream file contents."""
    keys = {
        "mesh": ("geometry", "mesh"),
        "solver": ("flight", "physics", "thermal", "solver", "geometry"),
        "temperature_export": ("geometry", "temperature_export", "flight", "physics", "thermal"),
        "aero": ("geometry", "flight", "physics", "thermal", "aero"),
        "calibration": ("geometry", "flight", "physics", "thermal", "solver", "calibration"),
        "thermal": ("geometry", "flight", "physics", "thermal", "temperature_export"),
    }[stage]
    values = {key: copy.deepcopy(cfg.get(key, {})) for key in keys}
    if "geometry" in values:
        values["geometry"].pop("input_file", None)
    if stage == "thermal":
        from .thermal import AEROTHERMAL_MODEL_VERSION
        values["mechanical"] = thermal_cfg
        if thermal_cfg.get("loading", {}).get("mode") == "aerodynamic_surface_load":
            values["mechanical_model_version"] = AEROTHERMAL_MODEL_VERSION
    if stage == "aero":
        from .aero import MODEL_VERSION
        values["export_model_version"] = MODEL_VERSION
    if stage == "calibration":
        from .aerothermal_calibration import MODEL_VERSION, calibration_settings
        values["surface_load_model_version"] = MODEL_VERSION
        values["effective_calibration_settings"] = calibration_settings(cfg)
    inputs = {"geometry": file_hash(cfg["geometry"]["input_file"])}
    aerodynamic = thermal_cfg.get("loading", {}).get("mode") == "aerodynamic_surface_load"
    if stage == "thermal" and not aerodynamic:
        from .temperature_export import temperature_paths
        inputs["wall_nodes"] = file_hash(temperature_paths(cfg)[0].parent / "wall_temperature_nodes.csv")
    dependencies = {"mesh": (), "solver": ("mesh",), "temperature_export": ("solver",),
                    "aero": ("solver",), "calibration": ("mesh", "solver", "temperature_export"),
                    "thermal": ("solver", "temperature_export", "calibration") if aerodynamic else ("solver", "temperature_export")}[stage]
    for dependency in dependencies:
        inputs[dependency] = {str(path.name): file_hash(path) for path in outputs[dependency]}
    return hashlib.sha256(json.dumps(dict(version=2, settings=values, inputs=inputs),
                                       sort_keys=True, allow_nan=False).encode()).hexdigest()


def output_hashes(paths):
    return {str(path): file_hash(path) for path in paths}


def validate_stage_outputs(stage, paths):
    """A nonempty file is insufficient evidence of a usable exported load."""
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        raise BatchError(f"{stage} required output files are missing or empty")
    if stage == "temperature_export":
        from .reporting import read_csv
        try:
            rows = read_csv(paths[0])
            summary = read_json(paths[1])
            if (not rows or len(rows) != summary["face_count"] or summary.get("status") != "complete"
                    or summary.get("thermal_mode") not in ("adiabatic","radiative_equilibrium")):
                raise ValueError("Invalid solved-temperature export summary")
            for row in rows:
                for key in ("wall_temperature_K", "near_wall_fluid_temperature_K", "segment_length_m"):
                    value = float(row[key])
                    if not math.isfinite(value) or value <= 0:raise ValueError(f"Invalid {key}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise BatchError(f"Invalid temperature_export output: {exc}") from exc
    elif stage == "calibration" and read_json(paths[2]).get("status") != "complete":
        raise BatchError("Calibration package does not report a complete result")
    elif stage == "thermal" and (read_json(paths[0]).get("loading_mode") not in ("convection","prescribed_temperature","aerodynamic_surface_load") or read_json(paths[0]).get("status") != "complete"):
        raise BatchError("Mechanical summary must report the selected temperature-driven loading")
    elif stage == "aero" and read_json(paths[0]).get("status") != "complete":
        raise BatchError("aero summary does not report a complete result")


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


def _activity_snapshot(paths):
    """Record native files without treating old files or directories as activity."""
    states = {}
    for path in paths:
        try:
            stat = path.stat()
            states[path] = (stat.st_size, stat.st_mtime_ns) if path.is_file() else None
        except OSError:
            states[path] = None
    return states


def run_stage(command, log_path, hard_timeout_s=7200, inactivity_timeout_s=1200,
              *, activity_files=()):
    """Capture output and native file progress; stop this subprocess tree on timeout.

    MAPDL can write results and its native console file while its Python client is
    blocked in a command. Only changes to the explicitly watched files count as
    activity. The hard deadline is unaffected by either output stream.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    activity_paths = tuple(dict.fromkeys(resolve(REPO_ROOT, path) for path in activity_files))
    file_states = _activity_snapshot(activity_paths)
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1,
                               start_new_session=os.name != "nt")
    messages = queue.Queue()

    def collect():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=collect, daemon=True).start()
    started = last_activity = time.monotonic()
    last_native_activity = None
    last_activity_message = float("-inf")
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
                        last_activity = time.monotonic()
                except queue.Empty:
                    pass
                now = time.monotonic()
                current_states = _activity_snapshot(activity_paths)
                changed = [(path, state) for path, state in current_states.items()
                           if state is not None and state != file_states[path]]
                file_states = current_states
                if changed:
                    last_activity = now
                    path, state = changed[-1]
                    last_native_activity = (path, state[0], now)
                    if now - last_activity_message >= 10.0:
                        message = f"[study] Native activity: {path} ({state[0]} bytes)\n"
                        print(message, end="", flush=True)
                        log.write(message)
                        log.flush()
                        last_activity_message = now
                if now - started > hard_timeout_s or now - last_activity > inactivity_timeout_s:
                    reason = "hard" if now - started > hard_timeout_s else "inactivity"
                    detail = ""
                    if last_native_activity is not None:
                        path, size, observed = last_native_activity
                        detail = (f"; last native activity: {path} ({size} bytes), "
                                  f"{now - observed:.1f} s ago")
                    log.write(f"\nStage {reason} timeout{detail}\n")
                    log.flush()
                    raise BatchError(f"Stage {reason} timeout{detail}; see {log_path}")
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
    preserve_cold = False
    cold_directory = work / "cfd/cold_calibration"
    if stage == "calibration":
        from run_calibration import reusable_cold_export
        from .aerothermal_calibration import calibration_cfg
        preserve_cold = reusable_cold_export(cfg, calibration_cfg(cfg), cold_directory / "_batch/config.yaml")
    candidates = []
    for item in INVALIDATES[stage]:
        candidates.extend(paths[item])
        candidates.append(work / "batch_logs" / f"{item}.log")
        state["stages"].pop(item, None)
    if stage == "mesh":
        candidates.extend([work / "01_labeled_geometry.pmdb", work / "geometry_summary.json"])
    if stage in ("mesh", "solver"):
        candidates.extend([work / "03_setup.cas.h5", work / "summary.json", work / "body_wall_heat_flux.csv",
                           work / "solver_temperature_history.csv", work / "solver_temperature_history.out", work / "solver_diagnostics",
                           work / "solver_force_history.csv", work / "solver_force_history.out"])
    if "aero" in INVALIDATES[stage]:candidates.append(work / "aero_forces_drag.txt")
    if "temperature_export" in INVALIDATES[stage]:
        from .temperature_export import temperature_paths
        candidates.append(temperature_paths(cfg)[0].parent)
        candidates.append(work / "reports")
    if stage in ("mesh", "solver"):
        # Retain the superseded isothermal/solid workflow along with its CFD.
        candidates.extend([work / "thermal_bc", work / "thermal_analysis",
                           work / "batch_logs/thermal_export.log", work / "batch_logs/thermal.log"])
        if cfg.get("workflow",{}).get("strategy") == "single_solve":candidates.append(work/"cfd")
        state["stages"].pop("thermal_export", None)
        state["stages"].pop("thermal", None)
    if "calibration" in INVALIDATES[stage]:
        candidates.append(work / "cfd/cold_calibration")
    if "thermal" in INVALIDATES[stage]:
        # Per-nose figures must not continue to show a superseded rear-boundary
        # model if the new calibration/thermal stage fails. Gas figures can be
        # regenerated from their saved exports without another native solve.
        candidates.extend([work / "thermal_analysis", work / "reports"])
    if preserve_cold:
        # A thermal mask/package change does not invalidate the native cold
        # solve. Its verifier still checks configuration, hashes and residuals.
        candidates = [path for path in candidates if path != cold_directory and cold_directory not in path.parents]
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
    if cfg.get("workflow",{}).get("strategy") == "single_solve":
        from .gas_reporting import COMPARISON_ARTIFACTS
        for name in COMPARISON_ARTIFACTS:(work.parent/"comparison"/name).unlink(missing_ok=True)
        if (work.parent/"comparison/solid").is_dir():shutil.rmtree(work.parent/"comparison/solid")


def run_case(config_path, thermal_config_path="thermal_analysis.yaml", resume=True, stop_after=None,
             *, _runner=None, _enabled_stages=None, _timeouts=None, _allow_adopt=True):
    from .config import load_config
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    if cfg.get("workflow",{}).get("strategy") == "two_endpoint":
        from .endpoint_workflow import run_case as run_endpoints
        return run_endpoints(config_path,thermal_config_path,resume,stop_after,runner=_runner,
                             enabled=_enabled_stages,timeouts=_timeouts,allow_adopt=_allow_adopt)
    single = cfg.get("workflow",{}).get("strategy") == "single_solve"
    stop_after = stop_after or ("thermal" if single else "aero")
    work = Path(cfg["project"]["work_dir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if stop_after not in STAGES:
        raise BatchError(f"Unknown stop-after stage: {stop_after}")
    thermal_cfg = {}
    if stop_after == "thermal":
        from .convection_loading import validate_settings
        thermal_config_path = Path(thermal_config_path).resolve()
        thermal_cfg = load_yaml(thermal_config_path)
        validate_settings(cfg, thermal_cfg)
        if (thermal_cfg.get("loading", {}).get("mode") == "aerodynamic_surface_load"
                and not cfg.get("calibration", {}).get("enabled", False)):
            raise BatchError("aerodynamic_surface_load requires calibration.enabled: true in the case config; run study.py update to migrate it")
    enabled = set(STAGES if _enabled_stages is None else _enabled_stages)
    files = stage_files(work, cfg)
    state_path = work / "_batch/stage_state.json"
    state = read_json(state_path) or {"version": 2, "stages": {}}
    if single:state.update(version=4,strategy="single_solve")
    state.setdefault("stages", {})
    arguments = ["--config", str(config_path)]
    commands = {
        "mesh": ["run_all.py", *arguments, "--stop-after", "mesh"],
        "solver": ["run_solver_only.py", *arguments],
        "temperature_export": ["export_temperature_only.py", *arguments],
        "aero": ["export_aero_only.py", *arguments],
        "calibration": ["run_calibration.py", *arguments],
        "thermal": ["run_thermal_analysis.py", *arguments, "--thermal-config", str(thermal_config_path)],
    }
    started, outcomes = time.monotonic(), {}
    for stage in STAGES[:STAGES.index(stop_after) + 1]:
        if stage == "calibration" and not cfg.get("calibration", {}).get("enabled", False):
            continue
        fingerprint = stage_fingerprint(stage, cfg, thermal_cfg, files)
        record = state["stages"].get(stage)
        try:
            validate_stage_outputs(stage, files[stage])
            if single and stage == "temperature_export":
                from .gas_reporting import temperature_complete
                if not temperature_complete(cfg):raise BatchError("Temperature export does not match the selected radiative case")
            if single and stage == "thermal":
                from .mechanical_reporting import mechanical_complete
                if not mechanical_complete(work,cfg):raise BatchError("Mechanical results are stale or use a different load model")
            if stage == "calibration":
                from .aerothermal_calibration import verify_surface_package
                try:verify_surface_package(cfg)
                except (ValueError, OSError, RuntimeError) as exc:raise BatchError(str(exc)) from exc
            if stage == "aero":
                from .gas_reporting import aero_complete_for_report
                if not aero_complete_for_report(cfg):raise BatchError("Aerodynamic export is stale")
            valid_outputs = True
        except BatchError:
            valid_outputs = False
        if resume and valid_outputs and reusable(record, fingerprint, files[stage]):
            if stage == "solver":
                from .solution_quality import saved_quality
                quality = saved_quality(cfg)
                if quality["status"] == "failed" or (single and quality["status"] != "residual_screen_passed"):valid_outputs = False
        if resume and valid_outputs and reusable(record, fingerprint, files[stage]):
            if stage == "aero":
                from .aero import require_force_stability
                try:require_force_stability(cfg, read_json(files[stage][0]))
                except ValueError as exc:raise BatchError(str(exc)) from exc
            print(f"[resume] {cfg['project']['name']}: {stage} unchanged")
            outcomes[stage] = "reused"
            continue
        can_adopt = resume and _allow_adopt and record is None and valid_outputs and stage not in ("solver", "thermal", "calibration")
        if can_adopt and stage == "mesh" and Path(cfg["geometry"]["input_file"]).stat().st_mtime_ns > files[stage][0].stat().st_mtime_ns:
            can_adopt = False
        if stage == "aero" and can_adopt:
            from .trade_study import aero_complete
            can_adopt = aero_complete(work, cfg)
        if stage == "temperature_export" and can_adopt:
            from .gas_reporting import temperature_complete
            can_adopt = temperature_complete(cfg)
        if can_adopt:
            if stage == "aero":
                from .aero import require_force_stability
                try:require_force_stability(cfg, read_json(files[stage][0]))
                except ValueError as exc:raise BatchError(str(exc)) from exc
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
            activity = {}
            if stage == "thermal":
                activity["activity_files"] = [work / "thermal_analysis" / name for name in (
                    "nose_thermal.out", "nose_thermal.err", "nose_thermal.rth", "nose_thermal.rst",
                    "nose_thermal.db", "mapdl_console.out", "mapdl_commands.log", ".__tmp__.out")]
            runner([sys.executable, "-u", str(REPO_ROOT / commands[stage][0]), *commands[stage][1:]],
                   work / "batch_logs" / f"{stage}.log",
                   hard_timeout_s=float(timeout.get("hard_s", 7200)),
                   inactivity_timeout_s=float(timeout.get("inactivity_s", 1200)), **activity)
            validate_stage_outputs(stage, files[stage])
            if stage == "solver":
                from .solution_quality import assess_log, save_diagnostics
                from .utils import atomic_json
                quality = assess_log(work / "batch_logs/solver.log",cfg)
                quality["source_sha256"] = {"final_case":file_hash(files[stage][0]),"final_data":file_hash(files[stage][1])}
                summary = read_json(work/"summary.json")
                summary["numerical_quality"] = quality
                summary["convergence_status"] = quality["status"]
                atomic_json(work/"summary.json",summary)
                save_diagnostics(work / "batch_logs/solver.log",cfg)
                if quality["status"] == "failed" or (single and quality["status"] != "residual_screen_passed"):
                    raise BatchError("CFD did not converge; saved diagnostic case/data retained. " + "; ".join(quality.get("reasons",[quality.get("reason","Residuals not verified")])) )
            if single and stage == "temperature_export":
                from .gas_reporting import temperature_complete
                if not temperature_complete(cfg):raise BatchError("Exported temperatures are stale or incompatible with radiative equilibrium")
            if single and stage == "thermal":
                from .mechanical_reporting import mechanical_complete
                if not mechanical_complete(work,cfg):raise BatchError("Mechanical results lack current provenance")
            if stage == "calibration":
                from .aerothermal_calibration import verify_surface_package
                verify_surface_package(cfg)
            if stage == "aero":
                from .aero import require_force_stability
                require_force_stability(cfg, read_json(files[stage][0]))
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
            changed_keys = {key for key in ("geometry", "mesh", "flight", "physics", "thermal", "solver", "calibration",
                                            "aero", "temperature_export", "reporting") if old.get(key) != cfg.get(key)}
            affected = set()
            for key in changed_keys:
                first = ("mesh" if key in ("geometry", "mesh") else "solver" if key in ("flight", "physics", "thermal", "solver")
                         else "temperature_export" if key in ("temperature_export", "reporting") else "calibration" if key == "calibration" else "aero")
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
    stop_after = batch.get("stop_after", "thermal" if batch.get("run_thermal", False) else "aero")
    thermal_path = resolve(root, batch.get("thermal_config", "thermal_analysis.yaml"))
    from .endpoint_workflow import STAGES as ENDPOINT_STAGES
    endpoint = base.get("workflow",{}).get("strategy") == "two_endpoint"
    enabled = set(ENDPOINT_STAGES if endpoint else STAGES)
    if not batch.get("run_cfd", True):
        enabled -= {"mesh", "cold_wall", "adiabatic"} if endpoint else {"mesh", "solver", "calibration"}
    rows = []
    for geometry, name in zip(geometries, names):
        directory, started = runs / name, time.monotonic()
        config_path = directory / "_batch/config.yaml"
        write_case_config(base, geometry, name, directory, config_path, batch.get("force_fluent_gui", False))
        try:
            result = run_case(config_path, thermal_path, resume=batch.get("resume", True), stop_after=stop_after,
                              _enabled_stages=enabled, _timeouts=batch.get("timeouts"))
            from .gas_reporting import build_gas_row
            row = build_gas_row(cfg=load_yaml(config_path), status=result["status"], elapsed_s=result["elapsed_s"])
        except Exception as exc:
            print(f"[batch] {name} failed: {exc}", flush=True)
            from .gas_reporting import build_gas_row
            row = build_gas_row(cfg=load_yaml(config_path), status="failed", elapsed_s=time.monotonic() - started, error=str(exc))
        rows.append(row)
        save_batch_summary(runs / "comparison", rows)
        if row["status"] in ("failed", "unconverged") and batch.get("stop_on_error", False):
            break
    if batch.get("comparison_outputs", True):
        from compare_nose_trade import compare
        compare(runs, runs / "comparison")
    return 1 if any(row["status"] in ("failed", "unconverged") for row in rows) else 0
