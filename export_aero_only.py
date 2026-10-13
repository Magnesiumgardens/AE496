"""Extract forces from an existing solved case without remeshing or iterating."""
from pathlib import Path
import argparse
import json

from pipeline.aero import aerodynamic_summary, validate_saved_conditions
from pipeline.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.get("solver", {}).get("run_iterations", True):
        parser.error("run_iterations=false cannot establish a solved CFD result")
    work = Path(cfg["project"]["work_dir"])
    case = work / "04_final.cas.h5"
    data = work / "04_final.dat.h5"
    if not case.is_file() or not data.is_file():
        parser.error(f"A solved case/data pair is required: {case}, {data}")
    output = work / "aero_summary.json"
    import ansys.fluent.core as pyfluent

    runtime = cfg.get("runtime", {})
    logs = work / "aero_logs"
    logs.mkdir(parents=True, exist_ok=True)
    solver = None
    try:
        solver = pyfluent.launch_fluent(
            mode=pyfluent.FluentMode.SOLVER, dimension=pyfluent.Dimension.TWO,
            precision=pyfluent.Precision.DOUBLE,
            processor_count=int(runtime.get("processors", 4)),
            ui_mode=pyfluent.UIMode.NO_GUI, cwd=str(logs),
            start_timeout=int(runtime.get("fluent_start_timeout_s", 300)),
        )
        solver.settings.file.read_case_data(file_name=str(case))
        validate_saved_conditions(solver, cfg)
        base = cfg["geometry"].get("body_wall_name", "body_wall")
        zones = [z for z in solver.settings.setup.boundary_conditions.wall.get_object_names()
                 if z == base or z.startswith(base + ".")]
        result = aerodynamic_summary(solver, cfg, zones)
        result["source_case"] = str(case)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[aero] Drag: {result['drag_per_unit_span_N_m']:.6g} N/m")
        print(f"[aero] Wrote {output}")
    except Exception as exc:
        output.write_text(json.dumps({"status": "failed", "error": str(exc)}, indent=2), encoding="utf-8")
        raise
    finally:
        if solver is not None:
            solver.exit()


if __name__ == "__main__":
    main()
