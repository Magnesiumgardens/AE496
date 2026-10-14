"""Continue a verified saved steady case without resetting its flow field."""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import yaml

from .aero import validate_saved_conditions
from .cfd import launch_solver
from .iteration_history import iterate_with_history
from .reporting import read_csv, read_json
from .solver_numerics import configure_numerics
from .temperature_export import atomic_csv, temperature_conditions
from .utils import atomic_json, file_sha256, work_paths


def source_hashes(work):
    return {"final_case":file_sha256(work/"04_final.cas.h5"),
            "final_data":file_sha256(work/"04_final.dat.h5")}


def preflight(cfg):
    from .study import BatchError, stage_files, stage_fingerprint
    work = work_paths(cfg)["work"]
    if not cfg["solver"].get("run_iterations",True):raise BatchError("Continuation requires solver.run_iterations=true")
    for name in ("02_mesh.msh.h5","04_final.cas.h5","04_final.dat.h5"):
        if not (work/name).is_file() or (work/name).stat().st_size == 0:
            raise BatchError(f"Continuation requires a nonempty saved mesh/case/data: {work/name}")
    summary = read_json(work/"summary.json")
    source = source_hashes(work)
    quality = summary.get("numerical_quality",{})
    recorded_source = quality.get("source_sha256",summary.get("temperature_history",{}).get("source_sha256"))
    if recorded_source != source or summary.get("thermal_mode") != "adiabatic":
        raise BatchError("Saved CFD provenance is missing or changed; cannot safely continue this case")
    if quality.get("recent_temperature_or_pressure_limiting"):
        raise BatchError("Saved flow still has temperature/pressure limiting; stabilize a fresh solve before continuing")
    state = read_json(work/"_batch/stage_state.json")
    record = state.get("stages",{}).get("solver",{})
    context_matches = (summary.get("source_conditions") == temperature_conditions(cfg)
                       and summary.get("source_mesh_sha256") == file_sha256(work/"02_mesh.msh.h5")
                       and summary.get("source_geometry_sha256") == file_sha256(cfg["geometry"]["input_file"]))
    if not context_matches and record.get("fingerprint") != stage_fingerprint("solver",cfg,{},stage_files(work,cfg)):
        raise BatchError("Saved solution settings/mesh do not match this config; continuation would use a different case")
    return summary, state


def continue_solver(cfg, iterations, output_dir):
    """Write to a new directory, keeping the original solution intact on errors."""
    original = work_paths(cfg)["work"]
    output = Path(output_dir).resolve()
    output.relative_to(original)  # this operation owns only files inside this case
    if (output/"04_final.cas.h5").exists() or (output/"04_final.dat.h5").exists():
        raise ValueError("Continuation output must be a fresh directory")
    output.mkdir(parents=True,exist_ok=True)
    summary = read_json(original/"summary.json")
    segment_cfg = copy.deepcopy(cfg)
    segment_cfg["project"]["work_dir"] = str(output)
    segment_cfg["solver"]["max_iterations"] = int(iterations)
    solver = launch_solver(cfg,"continuation_logs")
    try:
        print(f"[continue] Loading saved solution: {original/'04_final.cas.h5'}",flush=True)
        solver.settings.file.read_case_data(file_name=str(original/"04_final.cas.h5"))
        validate_saved_conditions(solver,cfg)
        if solver.settings.setup.general.solver.time.get_state() != "steady":
            raise ValueError("Continuation is restricted to steady CFD")
        summary["numerics"] = configure_numerics(solver,cfg)
        try:
            print("[continue] Mesh quality report:",flush=True)
            solver.settings.mesh.quality()
        except (AttributeError, RuntimeError) as exc:
            print(f"[continue] Optional quality report unavailable: {exc}",flush=True)
        # Do not read_mesh, standard_initialize, hybrid_initialize, or overwrite
        # farfield/wall settings: retain the developed solution and residual scale.
        print(f"[continue] Continuing for up to {iterations} additional steady iterations; no initialization",flush=True)
        summary["temperature_history"] = iterate_with_history(solver,segment_cfg,summary["body_wall_zones"])
        solver.settings.file.write_case_data(file_name=str(output/"04_final.cas.h5"))
        source = source_hashes(output)
        if summary["temperature_history"].get("status") == "complete":
            summary["temperature_history"]["source_sha256"] = source
        summary.pop("numerical_quality",None)
        summary.update(solution_status="additional_iterations_completed", convergence_status="not_verified",
                       source_conditions=temperature_conditions(cfg),
                       source_mesh_sha256=file_sha256(original/"02_mesh.msh.h5"),
                       source_geometry_sha256=file_sha256(cfg["geometry"]["input_file"]),
                       continuation=dict(additional_iteration_budget=int(iterations),
                                         initialization_performed=False, source_sha256=source_hashes(original)))
        atomic_json(output/"summary.json",summary)
        return summary
    finally:
        solver.exit()


def merge_history(original, staged, summary, archive):
    metadata = summary.get("temperature_history",{})
    old, new = original/"solver_temperature_history.csv", staged/"solver_temperature_history.csv"
    if metadata.get("status") != "complete" or not old.is_file():return
    try:
        old_metadata = read_json(original/"summary.json").get("temperature_history",{})
        if (old_metadata.get("csv_sha256") != file_sha256(old)
                or old_metadata.get("source_sha256") != source_hashes(original)):
            metadata["history_scope"] = "Current segment; previous CSV provenance was unavailable or changed"
            metadata["previous_history"] = str(archive/old.name)
            return
        previous, current = read_csv(old), read_csv(new)
        if previous and current and int(current[0]["iteration"]) > int(previous[-1]["iteration"]):
            atomic_csv(new,previous+current)
            metadata.update(sample_count=len(previous)+len(current),csv_sha256=file_sha256(new),
                            history_scope="Combined non-overlapping measured iteration records")
        else:
            metadata["history_scope"] = "Current continuation segment; native iteration labels overlap prior segment"
        metadata["previous_history"] = str(archive/old.name)
    except (OSError, KeyError, ValueError) as exc:
        metadata["history_merge_warning"] = str(exc)


def continue_case(config_path, iterations=4800, courant=None, *, _runner=None):
    from .config import load_config
    from .solution_quality import assess_log, save_diagnostics
    from .study import (BatchError, REPO_ROOT, _save_state, _archive_stage, run_stage,
                        stage_files, stage_fingerprint, output_hashes)
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    if isinstance(iterations,bool) or int(iterations) != iterations or iterations <= 0:
        raise BatchError("Additional iterations must be a positive integer")
    _, state = preflight(cfg)
    work = work_paths(cfg)["work"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = work/"_history"/(stamp+"_continue")
    archive.mkdir(parents=True)
    names = ("04_final.cas.h5","04_final.dat.h5","summary.json","solver_temperature_history.csv",
             "solver_temperature_history.out","batch_logs/solver.log","_batch/stage_state.json")
    for name in names:
        if (work/name).is_file():
            (archive/name).parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(work/name,archive/name)
    if (work/"solver_diagnostics").is_dir():shutil.copytree(work/"solver_diagnostics",archive/"solver_diagnostics")
    shutil.copy2(config_path,archive/"source_config.yaml")
    if courant is not None:
        from .config import positive
        positive("courant",courant)
        raw = yaml.safe_load(config_path.read_text())
        raw["solver"]["courant_number"] = float(courant)
        config_path.write_text(yaml.safe_dump(raw,sort_keys=False),encoding="utf-8")
        cfg = load_config(config_path)
    staged = work/"_continuation"/stamp
    staged.mkdir(parents=True)
    runner = _runner or run_stage
    print(f"[continue] Previous solution retained: {archive}",flush=True)
    try:
        runner([sys.executable,"-u",str(REPO_ROOT/"run_solver_only.py"),"--config",str(config_path),
                "--continue-saved","--iterations",str(int(iterations)),"--output-dir",str(staged)],
               staged/"solver.log",hard_timeout_s=7200,inactivity_timeout_s=1200)
        for name in ("04_final.cas.h5","04_final.dat.h5","summary.json"):
            if not (staged/name).is_file() or (staged/name).stat().st_size == 0:
                raise BatchError(f"Continuation produced no usable {name}; previous solution retained")
        if read_json(staged/"summary.json").get("thermal_mode") != "adiabatic":
            raise BatchError("Continuation summary is invalid; previous solution retained")
    except Exception:
        if courant is not None:shutil.copy2(archive/"source_config.yaml",config_path)
        raise
    summary = read_json(staged/"summary.json")
    quality = assess_log(staged/"solver.log",cfg)
    quality["source_sha256"] = source_hashes(staged)
    summary.update(numerical_quality=quality,convergence_status=quality["status"])
    if quality["numerical_state"] == "unstable":
        atomic_json(staged/"summary.json",summary)
        if courant is not None:shutil.copy2(archive/"source_config.yaml",config_path)
        diagnostic_cfg = copy.deepcopy(cfg)
        diagnostic_cfg["project"]["work_dir"] = str(staged)
        save_diagnostics(staged/"solver.log",diagnostic_cfg)
        return dict(case=cfg["project"]["name"],status="unstable",numerical_quality=quality,
                    archive=str(archive),work_dir=str(work),diagnostics=str(staged/"solver_diagnostics"),
                    message="Continued flow became unstable; previous case/data and configuration retained")
    merge_history(work,staged,summary,archive)
    atomic_json(staged/"summary.json",summary)
    transfers = [(staged/name,work/name) for name in names[:5] if (staged/name).is_file()]
    transfers.append((staged/"solver.log",work/"batch_logs/solver.log"))
    replaced=[]
    try:
        for source,target in transfers:
            target.parent.mkdir(parents=True,exist_ok=True)
            os.replace(source,target);replaced.append(target)
    except Exception:
        for target in replaced:
            backup = archive/target.relative_to(work)
            if backup.is_file():shutil.copy2(backup,target)
            else:target.unlink(missing_ok=True)
        if courant is not None:shutil.copy2(archive/"source_config.yaml",config_path)
        raise
    # Invalidate downstream products only after a new complete pair is saved.
    for stage in ("temperature_export","aero"):
        _archive_stage(stage,work,cfg,state)
    files = stage_files(work,cfg)
    state["stages"]["solver"] = dict(status="complete" if quality["status"] == "residual_screen_passed" else "failed",
        fingerprint=stage_fingerprint("solver",cfg,{},files),output_sha256=output_hashes(files["solver"]),
        provenance="continued_saved_solution", numerical_quality=quality["status"])
    _save_state(work/"_batch/stage_state.json",state)
    save_diagnostics(work/"batch_logs/solver.log",cfg)
    return dict(case=cfg["project"]["name"],status="complete" if quality["status"] == "residual_screen_passed" else "not_converged",
                numerical_quality=quality, archive=str(archive),work_dir=str(work),diagnostics=str(work/"solver_diagnostics"))
