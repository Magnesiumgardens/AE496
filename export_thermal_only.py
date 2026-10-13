"""Compatibility entrypoint; export wall heating and forces in one Fluent session."""
from pipeline.cfd import export_cli

if __name__ == "__main__":
    export_cli(thermal=True, aero=True)
