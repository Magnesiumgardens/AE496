"""Compare existing CFD/thermal outputs without launching any Ansys process."""
from pathlib import Path
import argparse
import csv
import math

from pipeline.trade_study import aero_complete, create_trade_outputs
from run_batch import build_result_row, load_yaml, read_json, save_batch_summary


def compare(runs_dir, output_dir):
    rows = []
    for directory in sorted(Path(runs_dir).iterdir()):
        if not directory.is_dir() or not (directory / "aero_summary.json").exists():
            continue
        config_path = directory / "_batch" / "config.yaml"
        if not config_path.is_file():
            print(f"[comparison] Skipping {directory.name}: missing case config")
            continue
        cfg = load_yaml(config_path)
        thermal = read_json(directory / "thermal_analysis" / "thermal_summary.json")
        fresh = aero_complete(directory, cfg)
        status = "complete" if fresh and thermal.get("status") == "complete" else "incomplete"
        row = build_result_row(Path(cfg["geometry"]["input_file"]), directory.name,
                               directory, status, 0)
        if not fresh:
            row["aero_status"] = "stale"
        # Peak over saved history is valid for either thermal loading model.
        # Do not infer missing heat-load/mass/limit metrics from an unknown model.
        history = directory / "thermal_analysis" / "temperature_history.csv"
        if thermal.get("status") == "complete" and history.is_file():
            with history.open(encoding="utf-8-sig", newline="") as f:
                records = list(csv.DictReader(f))
            maxima = [float(r["maximum_K"]) for r in records]
            if not maxima or not all(math.isfinite(v) for v in maxima):
                raise ValueError(f"Invalid temperature history in {history}")
            initial = thermal.get("analysis", {}).get("initial_temperature_k")
            if initial is not None:
                maxima.append(float(initial))
            row["peak_temperature_K"] = max(maxima)
        rows.append(row)
    save_batch_summary(Path(output_dir), rows)
    created = create_trade_outputs(Path(runs_dir), rows, Path(output_dir))
    print(f"[comparison] {len(rows)} cases; outputs: {output_dir}")
    return rows, created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/nose_trade_comparison"))
    args = parser.parse_args()
    compare(args.runs_dir.resolve(), args.output_dir.resolve())
