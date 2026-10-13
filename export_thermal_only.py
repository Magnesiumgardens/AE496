"""Compatibility entrypoint; export adiabatic temperatures and forces."""
from pipeline.cfd import export_cli

if __name__ == "__main__":
    export_cli(thermal=True, aero=True)
