"""One entry point for batch/single-case execution and saved-result reporting."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Resume one CFD → direct-flux thermal case")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", help="Use an existing authoritative case YAML")
    selection.add_argument("--geometry", help="Create a case config from one CAD file and the base template")
    run.add_argument("--base-config", default="config.yaml")
    run.add_argument("--runs-dir", default="runs")
    run.add_argument("--thermal-config", default="thermal_analysis.yaml")
    run.add_argument("--stop-after", choices=("mesh", "solver", "thermal_export", "aero", "thermal"), default="thermal")
    run.add_argument("--no-resume", action="store_true")
    batch = commands.add_parser("batch", help="Run all configured nose geometries sequentially")
    batch.add_argument("--batch-config", default="batch.yaml")
    report = commands.add_parser("report", help="Regenerate per-case and comparison outputs without Ansys")
    report.add_argument("--runs-dir", default="runs")
    report.add_argument("--output-dir", default="runs/comparison")
    report.add_argument("--reference-dir", help="Import the provided combined historical comparison CSVs")
    export = commands.add_parser("export", help="Export wall heating and forces from an existing solved case")
    export.add_argument("--config", required=True)
    export.add_argument("--inspect-fields", action="store_true")
    args = parser.parse_args()
    if args.command == "batch":
        from pipeline.study import main as batch_main
        return batch_main(args.batch_config)
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
