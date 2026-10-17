"""Build an RE/cold-wall load calibration on the unchanged primary CFD mesh."""
from pathlib import Path
import argparse
import sys

import yaml

from pipeline.aerothermal_calibration import (calibration_cfg, calibration_paths,
    verify_calibration_export, build_surface_package)
from pipeline.config import load_config, validate_config
from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.utils import atomic_json, file_sha256


def run_calibration(cfg):
    from pipeline.cfd import export_saved_case
    from pipeline.solution_quality import assess_log, save_diagnostics
    from pipeline.study import run_stage

    if not cfg.get("calibration", {}).get("enabled", False):
        raise ThermalAnalysisError("Cold-wall load calibration requires calibration.enabled: true")
    paths = calibration_paths(cfg)
    try:
        verify_calibration_export(cfg, "re")
    except (ValueError, OSError, ThermalAnalysisError):
        export_saved_case(cfg, thermal=False, aero=False)
    cold = calibration_cfg(cfg)
    validate_config(cold)
    directory = paths["cold_work"]
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "_batch/config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cold, sort_keys=False), encoding="utf-8")
    log = directory / "batch_logs/solver.log"
    run_stage([sys.executable, "-u", str(Path(__file__).resolve()), "--config",
               cfg["_config_path"], "--cold-solve"], log)
    quality = assess_log(log, cold)
    quality["source_sha256"] = {key: file_sha256(directory / name) for key, name in
                               (("final_case", "04_final.cas.h5"), ("final_data", "04_final.dat.h5"))}
    import json
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(numerical_quality=quality, convergence_status=quality["status"])
    atomic_json(summary_path, summary)
    save_diagnostics(log, cold)
    if quality["status"] != "residual_screen_passed":
        raise ValueError("Cold-wall calibration did not pass the residual screen; diagnostic pair retained")
    export_saved_case(cold, thermal=False, aero=False)
    return build_surface_package(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cold-solve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.cold_solve:
        from pipeline.solve import solve_case
        cold = calibration_cfg(cfg)
        validate_config(cold)
        print(solve_case(cold, calibration_paths(cfg)["root"] / "02_mesh.msh.h5"), flush=True)
    else:
        print(run_calibration(cfg), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
