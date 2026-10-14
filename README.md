# AE496 — hypersonic nose study for CDR

A planar 2-D Ansys 2025 R1 study: **mesh → one radiating-wall CFD solve → wall-temperature and aerodynamic exports → separate Mechanical APDL conduction → per-nose and comparison plots**.

## What the current model calculates

Fluent solves a steady compressible gas energy field with an external-radiation wall condition. The approximate local balance is `q_gas = emissivity × sigma × (Tw⁴ − T_environment⁴)`. Fluent solves the equilibrium surface temperature `Tw(s)`; it does not hold the wall at 300 K or set its heat flux to zero. External wall radiation does not require a volumetric gas-radiation model.

```yaml
workflow:
  strategy: single_solve
thermal:
  mode: radiative_equilibrium
  wall_temperature_k: 300.0  # reference metadata; not a prescribed CFD wall temperature
  emissivity: 0.90
  external_radiation_temperature_k: 3.0
```

The exported `wall_temperature_K` is **equilibrium surface temperature**, not adiabatic recovery temperature. Separately exported fluid-side static temperatures use wall-face sample coordinates; they are not measurements at a specified distance from the wall. Their interpretation depends on Fluent's sampling and the near-wall mesh.

Mechanical then **prescribes that steady CFD temperature distribution on the solid surface** and solves transient conduction through tungsten. This is an explicit temperature boundary condition: the surface steps to the imposed field when the load is applied, while the interior warms over physical time. It does not predict gradual surface warm-up. No convection coefficient is derived, and no additional convection, radiation or prescribed heat-flux load is applied on that temperature-constrained boundary. Temperature-constraint reactions are not currently exported as heat load.

This approximation was explicitly selected for the new single-solve model. The uploaded original CFD code contained the radiating-wall branch; the supplied historical Mechanical runner used heat-flux loading and did not establish a previously working prescribed-temperature coupling. See [the evidence and model differences](docs/ORIGINAL_WORKFLOW_COMPARISON.md).

## Update and run on Windows

The maintained Git branch is `codex/nose-trade-foundation`. From your existing AE496 folder:

```powershell
.\.venv\Scripts\python.exe study.py update
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe study.py run --config ".\runs\blunt_1\_batch\config.yaml" --stop-after solver
    if ($LASTEXITCODE -eq 0) {
        .\.venv\Scripts\python.exe study.py run --config ".\runs\blunt_1\_batch\config.yaml" --stop-after thermal
    }
}
```

The updater backs up source/configuration under `logs/source_backups/`, preserves CAD, meshes and simulation files, and selects the single radiating-wall strategy. It replaces the previous generated hybrid/native startup defaults with the conservative controls below; explicit custom numerical controls and iteration budgets, flight direction and exposure duration are retained. The changed solver settings start a fresh solve and archive previous solver outputs under `_history/`, while unchanged CAD and mesh settings reuse the mesh. Mechanical runs only after the new solution passes the numerical screen. Prior cold-wall/adiabatic outputs cannot satisfy a radiative-equilibrium export.

For an installation predating `study.py update`, bootstrap through authenticated Git:

```powershell
$src = Join-Path $env:TEMP ("AE496-source-" + [guid]::NewGuid().ToString("N"))
git clone --single-branch --branch codex/nose-trade-foundation https://github.com/Magnesiumgardens/AE496.git $src
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe "$src\scripts\update_project.py" --git-repo "$src" --root "."
}
```

Updates use an owned `.runtime/source_checkout` and Git `--ff-only`. For a case outside `runs/*/_batch`, use `study.py migrate --config "path\to\case.yaml" --thermal-config "path\to\thermal_analysis.yaml"` to back up and convert both input files. Omitting `--thermal-config` converts only the CFD configuration.

For new geometry, inspect the mesh before solving:

```powershell
.\.venv\Scripts\python.exe study.py run --geometry ".\geometry_batch\blunt_1.x_t" --stop-after mesh
.\.venv\Scripts\python.exe study.py run --geometry ".\geometry_batch\blunt_1.x_t" --stop-after thermal
```

Geometry is one XY fluid surface: outer farfield minus the vehicle profile. The body wall is its inner closed loop. SpaceClaim 251 is retained for this installation. Fluent's Prime 2-D meshing workflow uses a 3-D meshing host; the exported mesh and CFD calculation remain planar 2-D. The current nose points toward +X and flow travels toward **−X**. Inspect the shock and near-wall resolution; no automatic retry removes inflation layers.

Use `--stop-after aero` to finish CFD exports without Mechanical. Batch execution and offline reporting are:

```powershell
.\.venv\Scripts\python.exe study.py batch --batch-config batch.yaml
.\.venv\Scripts\python.exe study.py report --runs-dir runs
```

`batch.yaml` selects whether to run the solid stage. Drag and lift are sectional forces in **N/m of span** from the single radiating-wall solution. `Cd` and `Cl` require a justified common `aero.reference_area_per_unit_span_m`; a whole-vehicle or axisymmetric reference is not inferred.

## Mechanical settings and physical time

```yaml
loading:
  mode: prescribed_temperature
initial_temperature_k: 300.0
include_radiation: false
transient:
  duration_s: 200.0
```

Mechanical APDL reconstructs the solid boundary from actual exported CFD vertices. It interpolates their prescribed wall temperature along each boundary segment to all associated FE boundary nodes, including midside nodes. Mapping coordinates and imposed versus solved temperatures remain available in the saved data.

Tungsten conductivity, density and heat capacity belong to the solid stage. The supplied constant values—`k = 170 W/(m K)`, `rho = 19250 kg/m³`, `Cp = 130 J/(kg K)`—are configurable approximations requiring verification over the studied temperature range. The initial interior temperature is 300 K. Duration defaults to **200 seconds** and remains configurable. This calculation holds one flight condition and one imposed spatial surface field throughout the exposure; changing duration does not create a changing flight envelope.

Changing only material settings or exposure reuses valid CFD exports and reruns Mechanical. The standalone command is:

```powershell
.\.venv\Scripts\python.exe study.py thermal --config ".\runs\blunt_1\_batch\config.yaml"
```

If the run reached `[study] blunt_1: thermal` and then reported an inactivity timeout, update and resume that case:

```powershell
.\.venv\Scripts\python.exe study.py update
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe study.py run --config ".\runs\blunt_1\_batch\config.yaml" --stop-after thermal
}
```

Resume reuses valid saved CFD, temperature exports and aerodynamic results, then reruns the unfinished Mechanical stage. Keep resume enabled to retain this reuse. The thermal stage reports startup, geometry, meshing, solve and result-reading progress, and streams the native meshing and solve output. Its inactivity check also watches actual MAPDL output and result-file changes; a long native command can remain active without Python stdout. The separate hard deadline still applies. An inactivity timeout by itself does not identify whether MAPDL was busy, waiting for a license or stalled.

`thermal_analysis.yaml` accepts `mapdl_start_timeout_s: 120` to bound the initial MAPDL connection wait. Native output and command logs are saved in `runs/<case>/thermal_analysis/mapdl_console.out` and `mapdl_commands.log`; the stage log is `runs/<case>/batch_logs/thermal.log`. Inspect these logs if the retry stops, rather than restarting CFD.

## Convergence and continuation

The template uses the numerical setup from the supplied successful adiabatic run:

```yaml
solver:
  initialization: freestream
  courant_number: 0.3
  flow_scheme: first-order-upwind
  max_iterations: 4800
```

That run converged at iteration 3200. The later divergent radiating-wall run used the same CAD and mesh hashes, but hybrid initialization, native Courant 5 and second-order upwind. This change restores the successful startup controls while retaining the approved radiating-wall physics; it does not establish that the radiation case will converge. First-order upwind provides a conservative starting solution and adds numerical diffusion, so check scheme and mesh sensitivity before using the results for CDR.

The baseline residual targets are **1e-3 for flow and energy**, synchronized with Fluent's native stopping criteria. These targets are not a temperature error bound. Temperature and pressure limits are not raised to conceal divergence. Recent limiting, nonfinite residuals or a failed screen block downstream Mechanical loading.

For a stable, unconverged single solution, continue the saved field without initializing again:

```powershell
.\.venv\Scripts\python.exe study.py continue --config ".\runs\blunt_1\_batch\config.yaml" --iterations 4800
```

An optional `--courant` deliberately changes and records that control. Candidate case/data and previous results remain recoverable. Repeated limiting or oscillation requires checking numerics, mesh and physics; extending the iteration budget alone does not establish validity.

`solver_temperature_history.csv` is an optional **CFD convergence diagnostic indexed by Fluent iteration**. It is not a transient solid temperature history and is not a Mechanical load. Mechanical's temperature histories use **physical seconds**. Its whole-solid min/mean/max statistics include boundary nodes; the wall histories show an imposed surface step followed by a constant spatial field.

## Saved outputs and plots

```text
runs/<nose>/
  _batch/config.yaml
  02_mesh.msh.h5
  04_final.cas.h5 / 04_final.dat.h5
  summary.json
  temperature/
    wall_temperatures.csv         # wall-face and separate fluid-side temperatures
    wall_temperature_nodes.csv   # ordered actual vertices; Mechanical temperature input
    temperature_summary.json
    fluid_temperature_nodes.csv  # optional supported fluid-node samples
  aero_summary.json
  thermal_analysis/              # solid physical-time results, fields and animation
  reports/                       # equilibrium wall/gas profiles and maps
  solver_diagnostics/
  batch_logs/
  _history/
runs/comparison/
  radiative_equilibrium_temperature_summary.csv
  radiative_equilibrium_report.json
  solid/                         # prescribed-surface and whole-solid comparisons
```

Each nose gets normalized back → nose → back temperature profiles, wall and gas sample maps, optional CFD convergence curves, and separate Mechanical temperature fields and physical-time histories. Enable `visualization.make_gif` for an animation. Comparisons include equilibrium surface temperature versus sectional drag and solid conduction histories. Surface history/profile titles explicitly identify prescribed loading. This mode produces no derived-`h` or fabricated heat-flux plots.

Current reports verify mode, source case/data, temperature input and solid result hashes. Incompatible flight, gas, wall-radiation, mesh, material or exposure settings are excluded or separated. Previously generated calibration/adiabatic figures are removed from current report views; simulation source data remain retained.

Historical supplied CSVs remain available explicitly:

```powershell
.\.venv\Scripts\python.exe study.py report --reference-dir examples/reference_results --output-dir runs/historical_comparison
```

Historical data are reference fixtures, not newly validated radiative-equilibrium results. The earlier two-endpoint convection and direct-adiabatic/specified-`h` implementations remain compatibility paths for explicitly selected models. They are not required by the active single-solve configuration.

## Validation limits

Offline tests cover wall-condition adapters, sampling, prescribed-temperature mapping, numerical screening, source/result provenance, configuration migration, comparisons and Git updates:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The cloud environment has client libraries and offline plotting, but no licensed Fluent or Mechanical engines. Windows execution is still required to verify the live radiating-wall solve and FE conduction. For CDR, assess settled temperatures and forces, conservation, mesh/numerical sensitivity, high-temperature gas and tungsten properties, the planar approximation and the prescribed-surface idealization. Successful exports or a residual screen alone do not establish material survivability or an optimum nose.

A fresh setup uses Python 3.12 or later and `pip install -r requirements.txt`. Cloud setup uses `bash scripts/setup_cloud.sh`, then `source scripts/activate_cloud.sh`.
