# Migration from the previous project

Use the update and run commands in [README.md](README.md). Shared `pipeline/`
modules hold the implementation; `study.py` runs, resumes and reports the study.
The familiar solver, exporter, thermal and batch scripts remain compatibility
entry points.

Keep original CAD inputs in `geometry_batch/` and case folders in `runs/<case>/`.
The case-specific `_batch/config.yaml` is the authoritative work-directory and
geometry mapping. Updates preserve meshes and simulation files and retain
superseded results under `_history/`. A prior direct-adiabatic solution is not
silently adopted as a cold-wall calibration.

The restored default is **cold-wall steady CFD → warm-started adiabatic CFD →
spatial convection coefficient and recovery temperature → separate transient
Mechanical APDL analysis**. Aerodynamic exports and comparison reports are
retained. Cold-wall heat flux calibrates `h`; it is not applied as a constant
flux through the solid exposure. See the evidence and limits in
[docs/ORIGINAL_WORKFLOW_COMPARISON.md](docs/ORIGINAL_WORKFLOW_COMPARISON.md).

Review flight direction, explicit heat-flux sign, reference width, tungsten
properties, radiation and exposure duration on a known case before a batch.
The current coefficient is spatially varying but fixed as the solid warms;
wall-temperature dependence and a changing flight envelope require additional
calibration rather than a guessed correction.
