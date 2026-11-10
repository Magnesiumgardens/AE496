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
    run.add_argument("--stop-after", choices=("mesh", "cold_wall", "adiabatic", "thermal_bc", "solver", "temperature_export", "aero", "calibration", "thermal"))
    run.add_argument("--thermal-config", default="thermal_analysis.yaml")
    thermal = commands.add_parser("thermal", help="Run Mechanical with the selected calibrated surface load")
    thermal.add_argument("--config", required=True)
    thermal.add_argument("--thermal-config", default="thermal_analysis.yaml")
    run.add_argument("--no-resume", action="store_true")
    continuation = commands.add_parser("continue", help="Continue the saved CFD solution without reinitialization")
    continuation.add_argument("--config", required=True)
    continuation.add_argument("--iterations", type=int, default=4800)
    continuation.add_argument("--courant", type=float, help="Set and save a new Courant number for this case")
    continuation.add_argument("--endpoint",choices=("cold_wall","adiabatic"),default="cold_wall")
    continuation.add_argument("--thermal-config",default="thermal_analysis.yaml")
    continuation.add_argument("--stop-after", choices=("aero", "thermal"),
                              help="Finish CFD exports without rebuilding the thermal study when set to aero")
    batch = commands.add_parser("batch", help="Run all configured nose geometries sequentially")
    batch.add_argument("--batch-config", default="batch.yaml")
    report = commands.add_parser("report", help="Regenerate per-case and comparison outputs without Ansys")
    report.add_argument("--runs-dir", default="runs")
    report.add_argument("--output-dir", default="runs/comparison")
    report.add_argument("--reference-dir", help="Import the provided combined historical comparison CSVs")
    peak_flux = commands.add_parser("report-heat-flux", help="Compare peaks directly from existing surface heat-flux CSVs; no Ansys")
    peak_flux.add_argument("--runs-dir", default="runs")
    peak_flux.add_argument("--output-dir", help="Defaults to <runs-dir>/comparison/peak_heat_flux")
    peak_flux.add_argument("--geometry-dir", help="Select CAD cases from this folder; auto-detects geometry_batch beside runs")
    peak_flux.add_argument("--geometry-glob", default="*.x_t", help="CAD filename pattern, as in batch.yaml")
    peak_flux.add_argument("--cases", nargs="+", help="Case-insensitive CAD/case names or wildcard patterns, e.g. '*mk3*' '*mk4*'")
    internal = commands.add_parser("internal-compare", help="Solve internal heating from saved Mach-8 loads, or replot saved internal results")
    internal.add_argument("--runs-dir", default="runs")
    internal.add_argument("--screening-config", default="internal_screening.yaml")
    internal.add_argument("--geometry-dir", help="CAD selection folder; defaults to geometry_batch beside runs when present")
    internal.add_argument("--cases", nargs="+", help="Case-insensitive names or wildcards, e.g. '*mk3*' '*mk4*'")
    internal.add_argument("--depth-mode", choices=("axial_tip", "surface_normal"), help="Override the explicit probe convention in the screening YAML")
    internal.add_argument("--material-axes", nargs=2, choices=("x", "y", "z"), help="Material directions mapped to global X and Y, respectively")
    internal.add_argument("--output-dir")
    internal.add_argument("--processors", type=int, help="Override the MAPDL core/thread count for this invocation")
    internal.add_argument("--duration-s", type=float, help="Override the physical exposure duration, e.g. 170")
    internal.add_argument("--swap-labels", nargs=2, metavar=("CASE_A", "CASE_B"),
                          help="Swap two comparison display names without renaming source data")
    internal.add_argument("--report-only", action="store_true", help="Check receipts and regenerate plots without MAPDL")
    export = commands.add_parser("export", help="Export solved wall temperatures and forces from the selected CFD case")
    export.add_argument("--config", required=True)
    export.add_argument("--inspect-fields", action="store_true")
    force_status = commands.add_parser("force-status", help="Read force-history recording and verification details without Ansys")
    force_status.add_argument("--config", required=True)
    calibration_status = commands.add_parser("calibration-status", help="Inspect saved matched RE/cold heating anchors without Ansys")
    calibration_status.add_argument("--config", required=True)
    migrate = commands.add_parser("migrate", help="Back up and select the single radiative-equilibrium CFD workflow")
    migrate.add_argument("--config", required=True)
    migrate.add_argument("--thermal-config",help="Also back up and convert a separate Mechanical YAML")
    commands.add_parser("update", help="Fetch the study branch through Git and update with backups")
    args = parser.parse_args()
    if args.command == "internal-compare":
        from pipeline.internal_screening import compare_internal
        try:
            result = compare_internal(args.runs_dir, args.screening_config, geometry_dir=args.geometry_dir,
                cases=args.cases, output_dir=args.output_dir, depth_mode=args.depth_mode,
                material_axes=args.material_axes, report_only=args.report_only, processors=args.processors,
                swap_labels=args.swap_labels, duration_s=args.duration_s)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        return 0 if result["status"] == "complete" else 2
    if args.command == "calibration-status":
        import json
        from pipeline.config import load_config
        from pipeline.aerothermal_calibration import diagnose_calibration
        from pipeline.thermal_loading import ThermalAnalysisError
        try:
            result = diagnose_calibration(load_config(args.config))
        except (ValueError, OSError, ThermalAnalysisError) as exc:
            result = dict(status="unavailable", error=str(exc))
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] == "passed" else 2
    if args.command == "force-status":
        import json
        from pipeline.config import load_config
        from pipeline.force_history import force_history_diagnostics
        print(json.dumps(force_history_diagnostics(load_config(args.config)), indent=2))
        return 0
    if args.command == "update":
        from update import main as update_main
        return update_main()
    if args.command == "report-heat-flux":
        from pipeline.peak_flux_reporting import report_peak_heat_flux
        try:
            result = report_peak_heat_flux(args.runs_dir, args.output_dir, geometry_dir=args.geometry_dir,
                                          geometry_glob=args.geometry_glob, cases=args.cases)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        return 0 if result["case_count"] else 2
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
        run_case(args.config,args.thermal_config,stop_after=args.stop_after)
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
