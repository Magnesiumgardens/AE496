"""Compatibility command for the sequential study batch."""
import argparse

from pipeline.study import (BatchError, _failed_during_meshing, build_result_row,
                            load_yaml, main, read_json, run_case, save_batch_summary,
                            validate_thermal_inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--batch", "--batch-config", dest="config", default="batch.yaml")
    args = parser.parse_args()
    try:
        raise SystemExit(main(args.config))
    except (BatchError, ValueError, OSError) as exc:
        parser.exit(1, f"[batch] {exc}\n")
