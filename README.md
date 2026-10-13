# AE496 — planar hypersonic nose trade study

This branch is the starting point for a fresh Windows / Ansys 2025 R1 rebuild.
It contains the supplied solver, batch and thermal files plus new aerodynamic
reporting and comparison code. **It is not yet a complete runnable pipeline.**

## Available now

- Planar drag and lift per unit span, with explicit optional Cd/Cl normalization.
- Force export from completed Fluent case/data files.
- CSV/JSON summaries and drag-versus-temperature screening plots.
- Thermal-limit exposure, fixed-load heating and filled-section mass estimates.
- Unit tests using synthetic data and Fluent-interface fixtures.

Read [the installation and model notes](NOSE_TRADE_README.md) before using the
files with an existing project. You can download this branch from GitHub with
**Code → Download ZIP**, or clone it using the command shown by the Code menu.

## Rebuild work remaining

The uploaded files did not include `pipeline/config.py`, `geometry.py`,
`mesh.py`, `thermal_export.py`, `utils.py`, `run_all.py`, or the adiabatic-reference
entry point. They must be supplied or implemented before the full application
can run. A successful SpaceClaim backend launch has been verified on the user's
Windows machine; the standalone Geometry service is not installed there.

The provided thermal YAML requests endpoint-derived convection, whereas its
Python runner implements fixed heat flux only. The current checks reject that
mismatch rather than claiming the intended thermal model ran. Rebuilding the
endpoint thermal workflow remains required.

## Validation

17 local unit tests passed. No licensed Fluent/MAPDL simulation has been run
in the cloud. Convergence, mesh independence and material suitability are not
certified by these tests.

For the local numerical/reporting tests, install numpy, PyYAML and matplotlib,
then run from the repository root:

```text
python -m unittest discover -s tests -v
```

The Ansys workflow requires the appropriate PyAnsys packages, locally installed
Ansys products and licenses. The known client versions are PyFluent 0.42.1 and
PyAnsys Geometry 0.17.2.
