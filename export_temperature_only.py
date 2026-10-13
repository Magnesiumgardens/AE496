"""Export adiabatic wall/fluid-side temperatures from a saved CFD solution."""
from pipeline.cfd import export_cli

if __name__ == "__main__":
    export_cli(thermal=True, aero=False)
