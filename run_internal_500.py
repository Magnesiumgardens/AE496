"""Run the calibrated C/C MK3/MK4 solid transients for 500 physical seconds."""
import argparse
from pathlib import Path

from pipeline.internal_screening import compare_internal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true", help="Replot existing 500 s results without solving")
    parser.add_argument("--processors", type=int, default=1)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        result = compare_internal(root / "runs", root / "internal_screening_carbon_carbon.yaml",
            cases=["*mk3*", "*mk4*"], duration_s=500., processors=args.processors,
            report_only=args.report_only,
            swap_labels=["mk3_body_internals_base_nose1", "mk3_body_internals_blunt_nose"])
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
