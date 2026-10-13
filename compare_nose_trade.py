"""Rebuild individual nose reports and comparisons without starting Ansys."""
from pathlib import Path
import argparse

from pipeline.reporting import (build_result_row, create_comparison_outputs,
                                load_yaml, read_json, save_batch_summary)
from pipeline.trade_study import aero_complete


def compare(runs_dir, output_dir, reference_dir=None):
    if reference_dir is None:
        from pipeline.gas_reporting import create_gas_reports
        return create_gas_reports(runs_dir, output_dir)
    runs_dir, output_dir = Path(runs_dir), Path(output_dir)
    rows = []
    for directory in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        config_path = directory / "_batch/config.yaml"
        if not directory.is_dir() or not config_path.is_file():
            continue
        if not (directory / "aero_summary.json").is_file() and not (directory / "thermal_analysis/thermal_summary.json").is_file():
            continue
        cfg = load_yaml(config_path)
        thermal = read_json(directory / "thermal_analysis/thermal_summary.json")
        try:
            fresh = aero_complete(directory, cfg)
        except (ValueError, KeyError):
            fresh = False
        status = "complete" if fresh and thermal.get("status") == "complete" else "incomplete"
        row = build_result_row(Path(cfg["geometry"]["input_file"]), directory.name, directory, status, 0)
        if not fresh and row["aero_status"] == "complete":
            row["aero_status"] = "stale"
        rows.append(row)
    save_batch_summary(output_dir, rows)
    created = create_comparison_outputs(runs_dir, rows, output_dir, reference_dir)
    print(f"[report] {len(rows)} simulated cases; outputs: {output_dir}")
    return rows, created


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/comparison"))
    parser.add_argument("--reference-dir", type=Path, help="Import existing combined historical CSVs")
    args = parser.parse_args(argv)
    compare(args.runs_dir.resolve(), args.output_dir.resolve(), args.reference_dir)


if __name__ == "__main__":
    main()
