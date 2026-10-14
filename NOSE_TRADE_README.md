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

The conservative startup uses freestream initialization, Courant 0.3,
first-order upwind and a 4800-iteration budget, following the numerical controls
in the successful adiabatic run. Updates migrate the previous generated
hybrid/native defaults and retain explicit custom controls and budgets. Run
through `--stop-after solver` first; a changed solver setup starts fresh while
reusing unchanged mesh data. The radiation case still needs live verification,
and first-order results need scheme and mesh sensitivity checks for CDR.

After a thermal-stage inactivity timeout, update and resume the existing case
with `study.py run --config ".\runs\blunt_1\_batch\config.yaml" --stop-after thermal`.
Valid CFD and export stages are reused. Mechanical now reports its phases,
streams native meshing/solve output, and counts native output/result-file
changes as activity while retaining the hard deadline. The configurable
`mapdl_start_timeout_s` defaults to 120 seconds. Diagnostic logs are retained
under `runs/<case>/thermal_analysis/` and `batch_logs/thermal.log`; see the
[update and recovery commands](README.md#mechanical-settings-and-physical-time).

Review nose orientation, radiation environment, reference width, tungsten
properties and exposure before batching. Default airflow remains toward −X,
SpaceClaim remains selected, and the current default exposure is 200 seconds.
