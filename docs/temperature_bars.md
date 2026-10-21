# Maximum-temperature bar comparisons

The comparison command regenerates maximum-temperature bars from saved CFD and
Mechanical results. It does not start Fluent or Mechanical or rerun a batch.

From AE496 in PowerShell:

```powershell
.\.venv\Scripts\python.exe study.py update
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe study.py report --runs-dir ".\runs"
}
```

The raw steady-CFD comparison folder, normally `runs/comparison`, contains:

- `raw_max_wall_temperature_comparison.png` and its CSV: maximum exported CFD
  wall temperature for each nose. For the current radiative-equilibrium model,
  this is equilibrium wall temperature, not the transient tungsten temperature.
- `raw_max_near_wall_fluid_temperature_comparison.png` and its CSV: maximum
  exported fluid-side temperature sampled at the wall-face locations. These
  samples do not represent a specified distance from the wall or a maximum
  over the whole gas domain.

Only current validated temperature exports enter the raw bars. These plots
follow the existing CFD comparison-compatibility checks; incompatible flight,
wall-model, reference or mesh settings do not get combined into a
common gas comparison. CFD convergence iterations are not physical seconds.

The solid comparison folder, normally `runs/comparison/solid`, contains:

- `max_temperature_comparison.png` and its CSV: the highest solved temperature
  over all solid nodes and saved physical times, including the initial state.
- `final_temperature_comparison.png` and its CSV: the maximum solved solid
  temperature at the last saved physical time for each nose.

For a configured 200-second run, the first solid metric covers 0 through 200 s
and the second uses the result at 200 s. The peak may occur before the final
time, so these are separate metrics. Reports use the saved configured exposure;
they do not relabel a 400-second run as 200 seconds. Incompatible solid-analysis
conditions are reported in separate `conditions_*` subfolders.

All temperatures are in kelvin. Every bar is labeled by nose and accompanied
by CSV values. The raw CFD and physical-time solid quantities are separate
because they describe different states. Existing source-provenance and
Mechanical stored-energy checks remain required; regenerating graphs does not
make stale or physically inconsistent results eligible.
