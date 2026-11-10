"""Compatibility entry point for the current 500-second internal study."""
from run_internal_500 import main


if __name__ == "__main__":
    print("[internal] The convenience study now runs for 500 s; use run_internal_500.py.", flush=True)
    raise SystemExit(main())
