# Current nose-study workflow

Use the update/run commands in [README.md](README.md). Shared `pipeline/`
modules implement the study; `study.py` runs, resumes and regenerates reports.
The maintained Git branch is `codex/nose-trade-foundation`.

The primary CFD configuration remains `workflow.strategy: single_solve` and
`thermal.mode: radiative_equilibrium`. Fluent solves the equilibrium wall
temperature and exports sectional drag/lift. A separate cold-wall calculation
on the same fluid mesh supplies a second gas-heating calibration state. It
does not directly prescribe constant heat flux throughout the solid exposure.

The active Mechanical mode is `loading.mode: aerodynamic_surface_load`.
It solves surface and interior temperature from a uniform **300 K** initial
state. Native convection and radiation loads depend on the current surface
temperature in each nonlinear thermal iteration; no surface `D,TEMP`
constraint is applied in this mode.

At each matched CFD face:

```text
q_eq = epsilon × sigma × (T_eq⁴ − T_rad⁴)
H = (q_cold − q_eq) / (T_eq − T_cold)
B = q_cold + H × T_cold
q_net(Tw) = B − H × Tw − epsilon × sigma × (Tw⁴ − T_rad⁴)
```

`H` and `B/H` define an equivalent native convection load. `B/H` is a fitted
driving temperature, not directly sampled hot-air or adiabatic temperature.
The code obtains the RE gas-heating anchor from the verified wall radiation
balance because a generic Fluent wall heat-flux field is not reliably
identified as a separated convective flux on a Radiation wall.

CFD-face conductance/source fields are projected to actual boundary vertices.
Mechanical linearly interpolates conductance and driving temperature along
the reconstructed polygon boundary. This introduces spatial projection and
interpolation errors reported in the calibration summary. The two-anchor
affine heating law is an approximation and is extrapolated without clipping
outside the two calibration temperatures.

Keep CAD in `geometry_batch/` and case folders in `runs/<case>/`. Each case's
`_batch/config.yaml` remains authoritative. Updates preserve meshes, previous
simulation data, flow direction and explicit numerical controls. Current
compatible primary RE results can be reused, while new calibration and
Mechanical stages are generated as needed. Both calibration solutions must
pass matching residual screens; no adiabatic solve is required here.

The default physical duration is **200 s**. Set `transient.duration_s: 400.0`
in the selected thermal YAML for 400 s. Exposure/material changes invalidate
the solid results while retaining current CFD/calibration data. To retain
separate input settings, use `--thermal-config thermal_400.yaml`; rerunning
the same case archives invalidated thermal outputs under `_history/`.

`solver_temperature_history.csv` is a steady CFD **iteration** diagnostic.
`thermal_analysis/wall_temperature_history.csv` and `temperature_history.csv`
contain solved temperatures against **physical seconds**. New surface-load
histories separate convection into the solid, radiation out and net heat
input (W/m²), cumulative local energy (J/m²), and globally integrated boundary
power/energy per unit span (W/m, J/m). See
[the file/assumption reference](docs/aerodynamic_surface_load.md).

The conservative startup remains freestream initialization, Courant 0.3,
first-order upwind and a 4800-iteration budget. Airflow remains toward −X and
SpaceClaim remains selected. Drag/lift are from the primary RE solution and
are not updated during the solid transient. No AoA/Mach sweep, flow/solid
feedback, temperature-dependent tungsten properties or emissivity model is
introduced. Tungsten defaults remain constant `k=170 W/(m K)`,
`rho=19250 kg/m³`, `Cp=130 J/(kg K)`, with constant emissivity 0.9.

After a thermal-stage timeout, update and resume the existing case. The
runner watches native MAPDL output/result changes as well as console output
and retains a hard deadline. Diagnostic logs are in
`runs/<case>/thermal_analysis/mapdl_console.out`, `mapdl_commands.log` and
`batch_logs/thermal.log`. The configurable `mapdl_start_timeout_s` defaults
to 120 s. See [recovery commands](README.md#mechanical-settings-and-physical-time).

The previous prescribed-temperature and cold/adiabatic convection modes
remain explicit compatibility paths. Their old results retain their original
labels and are not accepted as new calibrated-load results. Offline tests
check algorithms, source provenance, load commands and reports. Native
Fluent/MAPDL execution still needs verification on the licensed Windows host.
