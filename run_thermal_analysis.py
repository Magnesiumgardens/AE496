"""Compatibility entry point; thermal model and data handling live in pipeline."""
from __future__ import annotations

import argparse

from pipeline.thermal import run_thermal_analysis
from pipeline.config import ConfigError
from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.convection_loading import validate_settings


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run MAPDL transient thermal analysis for a selected CFD case.")
    parser.add_argument("--config", default="config.yaml", help="Case-specific CFD YAML file.")
    parser.add_argument("--thermal-config", default="thermal_analysis.yaml")
    parser.add_argument("--run-dir", help="Compatibility option; must match the selected config's project.work_dir.")
    args = parser.parse_args(argv)
    try:
        return run_thermal_analysis(args.run_dir, args.config, args.thermal_config)
    except (ThermalAnalysisError, ConfigError) as exc:
        parser.exit(2, f"[thermal] {exc}\n")


if __name__ == "__main__":
    main()
