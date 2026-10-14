# Migration from the previous project

Use the update/run commands in [README.md](README.md). Shared `pipeline/`
modules implement the study; `study.py` runs, resumes and regenerates reports.
Familiar solver, exporter, thermal and batch scripts remain compatibility entry points.

The active strategy is **one radiating-wall steady CFD solve → solved equilibrium
surface-temperature export and aerodynamic forces → separate Mechanical APDL
conduction with prescribed surface temperature**. It requires no cold/adiabatic
pairing and derives no convection coefficient.

Mechanical imposes the steady spatial CFD wall temperature as a surface step.
The interior warms over configurable physical time; the surface warm-up is not
predicted. No extra radiation, convection or heat-flux load belongs on that
temperature-constrained boundary. The historical Mechanical heat-flux runner
did not implement this new coupling. Evidence and limits are recorded in
[docs/ORIGINAL_WORKFLOW_COMPARISON.md](docs/ORIGINAL_WORKFLOW_COMPARISON.md).

Keep CAD in `geometry_batch/` and case folders in `runs/<case>/`. Each case's
`_batch/config.yaml` remains authoritative. Updates preserve meshes, old
simulation files, flow direction and explicit numerical controls. A previous
cold-wall or adiabatic solution cannot satisfy the new radiating-wall model.
Current figures check source hashes and use equilibrium/prescribed-temperature
labels; historical reference data must be explicitly selected.

Review nose orientation, radiation environment, reference width, tungsten
properties and exposure before batching. Default airflow remains toward −X,
SpaceClaim remains selected, and the current default exposure is 200 seconds.
