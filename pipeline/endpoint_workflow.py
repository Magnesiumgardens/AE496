"""Two steady CFD endpoints, one shared mesh, then separate Mechanical analysis."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

from .reporting import read_json, load_yaml
from .utils import atomic_json, file_sha256

STAGES = ("mesh", "cold_wall", "cold_export", "adiabatic", "adiabatic_export", "thermal_bc", "aero", "thermal")
DEPENDENCIES = {"mesh":(), "cold_wall":("mesh",), "cold_export":("cold_wall",),
    "adiabatic":("cold_wall",), "adiabatic_export":("adiabatic",),
    "thermal_bc":("cold_export", "adiabatic_export"), "aero":("adiabatic",), "thermal":("thermal_bc",)}


def effective_cfg(cfg, kind):
    if kind not in ("cold_wall", "adiabatic"):raise ValueError(f"Unknown CFD endpoint: {kind}")
    result = copy.deepcopy(cfg)
    root = Path(cfg.get("_study_work_dir", cfg["project"]["work_dir"])).resolve()
    result.update(_endpoint_kind=kind, _study_work_dir=str(root))
    cold_thermal = cfg.get("_cold_wall_thermal",cfg["thermal"])
    if cold_thermal.get("mode") != "fixed_temperature":raise ValueError("Cannot recover the cold-wall calibration temperature from this configuration")
    result["_cold_wall_thermal"] = copy.deepcopy(cold_thermal)
    result["thermal"] = copy.deepcopy(cold_thermal)
    result["_study_solver"] = copy.deepcopy(cfg.get("_study_solver",cfg["solver"]))
    result["solver"] = copy.deepcopy(result["_study_solver"])
    result["project"]["work_dir"] = str(root/"cfd"/kind)
    if kind == "adiabatic":
        result["thermal"] = {"mode":"adiabatic"}
        result["solver"]["max_iterations"] = int(cfg.get("workflow", {}).get("adiabatic_reference_iterations", 600))
    else:
        # A prescribed 300 K wall history is constant and is not evidence of
        # heating or convergence. Residual diagnostics cover the cold solve.
        result["solver"].setdefault("temperature_history",{})["enabled"] = False
    result["solver"].update(cfg.get("workflow",{}).get(kind+"_solver",{}))
    directory = Path(cfg.get("temperature_export", {}).get("output_directory", "temperature"))
    result.setdefault("temperature_export", {})["output_directory"] = str(directory if directory.is_absolute() else root/directory)
    return result


def stage_files(cfg):
    from .temperature_export import temperature_paths
    work = Path(cfg["project"]["work_dir"])
    def endpoint(kind):return work/"cfd"/kind
    def solved(kind):return [endpoint(kind)/name for name in ("04_final.cas.h5", "04_final.dat.h5", "summary.json")]
    def exported(kind):return [endpoint(kind)/"thermal_bc"/name for name in ("wall_faces.csv", "wall_nodes.csv", "endpoint_summary.json")]
    csv_path, summary_path = temperature_paths(effective_cfg(cfg,"adiabatic"))
    return dict(mesh=[work/"02_mesh.msh.h5"], cold_wall=solved("cold_wall"), cold_export=exported("cold_wall"),
        adiabatic=solved("adiabatic"), adiabatic_export=exported("adiabatic")+[csv_path, summary_path, csv_path.parent/"wall_temperature_nodes.csv"],
        thermal_bc=[work/"thermal_bc/convection_nodes.csv", work/"thermal_bc/convection_summary.json"],
        aero=[work/"aero_summary.json"], thermal=[work/"thermal_analysis"/name for name in (
            "thermal_summary.json", "wall_temperature_history.csv", "thermal_nodes_final.csv")])


def fingerprint(stage, cfg, thermal_cfg, files):
    from .study import stage_fingerprint
    if stage == "mesh":return stage_fingerprint("mesh", cfg, {}, {"mesh":files["mesh"]})
    keys = {"cold_wall":("flight","physics","thermal","solver"), "adiabatic":("flight","physics","solver"),
        "cold_export":("thermal_export",), "adiabatic_export":("thermal_export","temperature_export"),
        "thermal_bc":("workflow","thermal"), "aero":("aero","flight","physics"), "thermal":()}[stage]
    values = {key:cfg.get(key,{}) for key in keys}
    if stage in ("cold_wall","adiabatic"):values["solver"] = effective_cfg(cfg,stage)["solver"]
    if stage == "adiabatic":values["reference_iterations"] = cfg["workflow"].get("adiabatic_reference_iterations",600)
    if stage == "thermal":values["mechanical"] = thermal_cfg
    inputs = {str(path):file_sha256(path) for dep in DEPENDENCIES[stage] for path in files[dep]}
    if stage in ("cold_wall","adiabatic"):
        inputs["geometry"] = file_sha256(cfg["geometry"]["input_file"])
        values["geometry"] = {k:v for k,v in cfg["geometry"].items() if k != "input_file"}
    return hashlib.sha256(json.dumps(dict(version=1, settings=values, inputs=inputs),sort_keys=True,allow_nan=False).encode()).hexdigest()


def downstream(stage):
    affected = {stage}
    for name in STAGES:
        if any(dep in affected for dep in DEPENDENCIES[name]):affected.add(name)
    return affected


def archive(stage, cfg, state):
    work = Path(cfg["project"]["work_dir"])
    files = stage_files(cfg)
    affected = downstream(stage)
    candidates = set(path for name in affected for path in files[name])
    for name in affected:
        state["stages"][name] = {"status":"stale", "reason":f"{stage} changed or failed"}
        candidates.add(work/"batch_logs"/(name+".log"))
    if "cold_wall" in affected:candidates.add(work/"cfd/cold_wall")
    if "adiabatic" in affected:candidates.add(work/"cfd/adiabatic")
    if "adiabatic_export" in affected:candidates.add(files["adiabatic_export"][3].parent)
    if "thermal_bc" in affected:candidates.add(work/"thermal_bc")
    if "thermal" in affected:candidates.add(work/"thermal_analysis")
    if any(name in affected for name in ("adiabatic_export","aero","thermal_bc","thermal")):candidates.add(work/"reports")
    existing = [p for p in candidates if p.exists()]
    parents = [p for p in existing if not any(q != p and q in p.parents for q in existing)]
    if not parents:return
    destination = work/"_history"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+stage)
    for path in parents:
        relative = path.resolve().relative_to(work.resolve())
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path),str(target))
    print(f"[resume] Previous outputs retained: {destination}",flush=True)
    # Comparison products describe multiple active sources and must be rebuilt
    # after invalidation; historical reference directories are untouched.
    from .gas_reporting import COMPARISON_ARTIFACTS
    comparison = work.parent/"comparison"
    for name in (*COMPARISON_ARTIFACTS,"nose_trade_summary.csv","drag_temperature_trade.png"):
        (comparison/name).unlink(missing_ok=True)
    if (comparison/"solid").is_dir():shutil.rmtree(comparison/"solid")


def validate(stage, cfg, files):
    from .study import BatchError, validate_stage_outputs
    if not all(p.is_file() and p.stat().st_size for p in files[stage]):raise BatchError(f"{stage}: required outputs missing or empty")
    if stage in ("cold_wall","adiabatic"):
        from .solution_quality import saved_quality
        quality = saved_quality(effective_cfg(cfg,stage))
        if quality.get("status") != "residual_screen_passed":
            raise BatchError(f"{stage} did not pass the residual screen: " + "; ".join(quality.get("reasons",[quality.get("reason","not verified")])))
    elif stage in ("cold_export","adiabatic_export"):
        from .endpoint_loading import _endpoint_metadata
        _endpoint_metadata(cfg,"cold_wall" if stage == "cold_export" else "adiabatic")
        if stage == "adiabatic_export":
            from .gas_reporting import temperature_complete
            if not temperature_complete(cfg):raise BatchError("Adiabatic temperature export is stale or invalid")
    elif stage == "thermal_bc":
        from .endpoint_loading import verify_convection_package
        verify_convection_package(cfg)
    elif stage == "thermal":
        if read_json(files[stage][0]).get("status") != "complete" or read_json(files[stage][0]).get("loading_mode") != "convective_from_cfd":
            raise BatchError("Mechanical endpoint-convection analysis is incomplete")
        from .mechanical_reporting import mechanical_complete
        if not mechanical_complete(Path(cfg["project"]["work_dir"]),cfg):raise BatchError("Mechanical results or provenance are stale")
    elif stage == "aero":validate_stage_outputs(stage,files[stage])


def run_case(config_path, thermal_config_path, resume=True, stop_after=None, *, runner=None, enabled=None, timeouts=None, allow_adopt=True):
    from .config import load_config
    from .study import BatchError, REPO_ROOT, run_stage, reusable, output_hashes
    cfg = load_config(config_path)
    stop_after = {"solver":"cold_wall", "temperature_export":"thermal_bc"}.get(stop_after,stop_after) or "thermal"
    if stop_after not in STAGES:raise BatchError(f"Unknown stop-after stage: {stop_after}")
    work = Path(cfg["project"]["work_dir"])
    work.mkdir(parents=True,exist_ok=True)
    thermal_cfg = load_yaml(thermal_config_path) if stop_after == "thermal" else {}
    if stop_after == "thermal":
        from .convection_loading import validate_settings
        validate_settings(cfg,thermal_cfg)
    files = stage_files(cfg)
    state_path = work/"_batch/stage_state.json"
    state = read_json(state_path) or {"stages":{}}
    state.update(version=3, strategy="two_endpoint")
    state.setdefault("stages",{})
    outcomes, started = {}, time.monotonic()
    for stage in STAGES[:STAGES.index(stop_after)+1]:
        signature = fingerprint(stage,cfg,thermal_cfg,files)
        record = state["stages"].get(stage)
        try:validate(stage,cfg,files); valid = True
        except (ValueError,OSError,RuntimeError):valid = False
        if resume and valid and reusable(record,signature,files[stage]):
            outcomes[stage]="reused"; print(f"[resume] {work.name}: {stage} unchanged",flush=True); continue
        if stage == "mesh" and resume and valid and allow_adopt and record is None and Path(cfg["geometry"]["input_file"]).stat().st_mtime_ns <= files[stage][0].stat().st_mtime_ns:
            state["stages"][stage] = dict(status="complete",fingerprint=signature,output_sha256=output_hashes(files[stage]),provenance="adopted_unverified")
            atomic_json(state_path,state); outcomes[stage]="adopted_unverified"; continue
        if enabled is not None and stage not in enabled:raise BatchError(f"{stage} is disabled and cannot be reused")
        archive(stage,cfg,state); atomic_json(state_path,state)
        print(f"[study] {work.name}: {stage}",flush=True)
        args = [sys.executable,"-u",str(REPO_ROOT/"run_endpoint.py"),"--config",str(Path(config_path).resolve()),
                "--stage",stage,"--thermal-config",str(Path(thermal_config_path).resolve())]
        limits = (timeouts or {}).get(stage,(timeouts or {}).get("solver",{}) if stage in ("cold_wall","adiabatic") else {})
        try:
            if stage == "cold_wall" and outcomes.get("mesh") == "executed":
                delay = min(60.,max(0.,float(cfg.get("runtime",{}).get("meshing_release_delay_s",0))))
                if delay:time.sleep(delay)
            (runner or run_stage)(args,work/"batch_logs"/(stage+".log"),
                hard_timeout_s=float(limits.get("hard_s",7200)), inactivity_timeout_s=float(limits.get("inactivity_s",1200)))
            if stage in ("cold_wall","adiabatic"):
                from .solution_quality import assess_log, save_diagnostics
                effective = effective_cfg(cfg,stage)
                quality = assess_log(work/"batch_logs"/(stage+".log"),effective)
                quality["source_sha256"] = {key:file_sha256(files[stage][i]) for i,key in enumerate(("final_case","final_data"))}
                summary = read_json(files[stage][2]); summary.update(numerical_quality=quality,convergence_status=quality["status"])
                atomic_json(files[stage][2],summary)
                save_diagnostics(work/"batch_logs"/(stage+".log"),effective)
            validate(stage,cfg,files)
        except Exception as exc:
            state["stages"][stage] = dict(status="failed",fingerprint=signature,error=str(exc))
            atomic_json(state_path,state); raise
        state["stages"][stage] = dict(status="complete",fingerprint=signature,output_sha256=output_hashes(files[stage]),provenance="executed")
        atomic_json(state_path,state); outcomes[stage]="executed"
    return dict(case=work.name,work_dir=str(work),status="complete",stages=outcomes,elapsed_s=round(time.monotonic()-started,3))


def execute_stage(cfg, stage, thermal_config):
    root = Path(cfg["project"]["work_dir"])
    if stage == "mesh":
        from .geometry import prepare_geometry
        from .mesh import build_mesh
        return build_mesh(cfg,prepare_geometry(cfg))
    if stage in ("cold_wall","adiabatic"):
        from .solve import solve_case
        return solve_case(effective_cfg(cfg,stage),root/"02_mesh.msh.h5",
                          reference_cfg=effective_cfg(cfg,"cold_wall") if stage == "adiabatic" else None)
    if stage in ("cold_export","adiabatic_export"):
        from .cfd import export_saved_case
        kind = "cold_wall" if stage == "cold_export" else "adiabatic"
        return export_saved_case(effective_cfg(cfg,kind), thermal=kind == "adiabatic", aero=False)
    if stage == "thermal_bc":
        from .endpoint_loading import build_convection_package
        return build_convection_package(cfg)
    if stage == "aero":
        from .cfd import export_saved_case
        return export_saved_case(effective_cfg(cfg,"adiabatic"),thermal=False,aero=True)
    if stage == "thermal":
        from .thermal import run_thermal_analysis
        return run_thermal_analysis(cfd_config=cfg["_config_path"],thermal_config=thermal_config)
    raise ValueError(f"Unknown endpoint stage: {stage}")
