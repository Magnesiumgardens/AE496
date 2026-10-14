"""One entry point for batch/single-case execution and saved-result reporting."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Resume one radiative-equilibrium CFD solve, aerodynamics and separate Mechanical")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", help="Use an existing authoritative case YAML")
    selection.add_argument("--geometry", help="Create a case config from one CAD file and the base template")
    run.add_argument("--base-config", default="config.yaml")
    run.add_argument("--runs-dir", default="runs")
    run.add_argument("--stop-after", choices=("mesh", "cold_wall", "adiabatic", "thermal_bc", "solver", "temperature_export", "aero", "thermal"))
    run.add_argument("--thermal-config", default="thermal_analysis.yaml")
    thermal = commands.add_parser("thermal", help="Run separate Mechanical APDL conduction with prescribed CFD wall temperatures")
    thermal.add_argument("--config", required=True)
    thermal.add_argument("--thermal-config", default="thermal_analysis.yaml")
    run.add_argument("--no-resume", action="store_true")
    continuation = commands.add_parser("continue", help="Continue the saved CFD solution without reinitialization")
    continuation.add_argument("--config", required=True)
    continuation.add_argument("--iterations", type=int, default=4800)
    continuation.add_argument("--courant", type=float, help="Set and save a new Courant number for this case")
    continuation.add_argument("--endpoint",choices=("cold_wall","adiabatic"),default="cold_wall")
    continuation.add_argument("--thermal-config",default="thermal_analysis.yaml")
    batch = commands.add_parser("batch", help="Run all configured nose geometries sequentially")
    batch.add_argument("--batch-config", default="batch.yaml")
    report = commands.add_parser("report", help="Regenerate per-case and comparison outputs without Ansys")
    report.add_argument("--runs-dir", default="runs")
    report.add_argument("--output-dir", default="runs/comparison")
    report.add_argument("--reference-dir", help="Import the provided combined historical comparison CSVs")
    export = commands.add_parser("export", help="Export solved wall temperatures and forces from the selected CFD case")
    export.add_argument("--config", required=True)
    export.add_argument("--inspect-fields", action="store_true")
    migrate = commands.add_parser("migrate", help="Back up and select the single radiative-equilibrium CFD workflow")
    migrate.add_argument("--config", required=True)
    migrate.add_argument("--thermal-config",help="Also back up and convert a separate Mechanical YAML")
    commands.add_parser("update", help="Fetch the study branch through Git and update with backups")
    args = parser.parse_args()
    if args.command == "update":
        from update import main as update_main
        return update_main()
    if args.command == "migrate":
        from pipeline.migration import migrate_config
        thermal_change=None
        if args.thermal_config:
            from datetime import datetime,timezone
            from pathlib import Path
            import shutil
            import yaml
            from pipeline.migration import single_solve_thermal_config
            path=Path(args.thermal_config).resolve()
            raw=yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(raw,dict):parser.error("Mechanical YAML must contain a mapping")
            changed=single_solve_thermal_config(raw)
            if changed != raw:thermal_change=(path,changed)
        # Read both inputs before modifying either, so a missing or malformed
        # Mechanical file does not leave the CFD model partly migrated.
        migrate_config(args.config)
        if thermal_change:
            path,changed=thermal_change
            stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            shutil.copy2(path,path.with_name(path.stem+"_before_single_solve_"+stamp+path.suffix))
            path.write_text(yaml.safe_dump(changed,sort_keys=False),encoding="utf-8")
        return 0
    if args.command == "batch":
        from pipeline.study import main as batch_main
        return batch_main(args.batch_config)
    if args.command == "continue":
        from pipeline.continuation import continue_case
        result = continue_case(args.config,args.iterations,args.courant,endpoint=args.endpoint)
        print(result)
        if result["status"] != "complete":
            print(f"[continue] {result.get("message", "Not converged yet; downstream analysis remains blocked")}. Diagnostics: {result["diagnostics"]}")
            return 2
        from pipeline.study import run_case
        run_case(args.config,args.thermal_config)
        from compare_nose_trade import compare
        from pathlib import Path
        work = Path(result["work_dir"])
        compare(work.parent,work.parent/"comparison")
        return 0
    if args.command == "thermal":
        from pipeline.thermal import run_thermal_analysis
        print(run_thermal_analysis(cfd_config=args.config, thermal_config=args.thermal_config))
        from compare_nose_trade import compare
        from pipeline.config import load_config
        from pathlib import Path
        work = Path(load_config(args.config)["project"]["work_dir"])
        compare(work.parent, work.parent / "comparison")
        return 0
    if args.command == "run":
        from pipeline.study import run_case
        config_path = args.config
        if args.geometry:
            from pathlib import Path
            from pipeline.study import safe_case_name, write_case_config
            from pipeline.reporting import load_yaml
            geometry = Path(args.geometry).resolve()
            if not geometry.is_file():
                parser.error(f"Geometry input does not exist: {geometry}")
            name = safe_case_name(geometry.stem)
            directory = Path(args.runs_dir).resolve() / name
            config_path = directory / "_batch" / "config.yaml"
            write_case_config(load_yaml(args.base_config), geometry, name, directory, config_path)
        result = run_case(config_path, args.thermal_config, resume=not args.no_resume, stop_after=args.stop_after)
        if result.get("status") == "complete" and (args.stop_after in ("aero", "thermal") or args.stop_after is None):
            from pipeline.config import load_config
            from pathlib import Path
            cfg = load_config(config_path)
            work = Path(cfg["project"]["work_dir"])
            from compare_nose_trade import compare
            compare(work.parent, work.parent / "comparison")
        print(result)
        return 0 if result.get("status") == "complete" else 2
    if args.command == "report":
        from compare_nose_trade import compare
        compare(args.runs_dir, args.output_dir, reference_dir=args.reference_dir)
        return 0
    from pipeline.config import load_config
    from pipeline.cfd import export_saved_case
    print(export_saved_case(load_config(args.config), inspect=args.inspect_fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
