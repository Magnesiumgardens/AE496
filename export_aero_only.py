"""Compatibility entrypoint for saved-case aerodynamic force export."""
from pipeline.cfd import export_cli

if __name__ == "__main__":
    export_cli(thermal=False, aero=True)
