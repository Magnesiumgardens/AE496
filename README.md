# AE496 — hypersonic nose study for CDR

A planar 2-D Ansys 2025 R1 study: **mesh → primary radiating-wall CFD + cold-wall calibration → temperature-dependent surface heating → transient Mechanical APDL conduction → per-nose and comparison plots**.

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

An auxiliary cold-wall CFD case on the **same mesh and flight condition** supplies signed gas heat flux at a configured uniform wall temperature, normally 300 K. It is a calibration state, not a fixed heat-flux load for the solid. The primary radiative-equilibrium case still supplies drag/lift and the equilibrium temperature `T_eq(s)`; `single_solve` remains the primary CFD strategy even though this thermal coupling requires the extra calibration solve.

At matched CFD faces the two gas-heating anchors are `q_cold` and `q_eq = epsilon × sigma × (T_eq⁴ − T_rad⁴)`. The code derives an affine approximation:

```text
H(s) = [q_cold(s) − q_eq(s)] / [T_eq(s) − T_cold]
B(s) = q_cold(s) + H(s) × T_cold
q_conv(s,Tw) = B(s) − H(s) × Tw
q_net(s,Tw) = q_conv(s,Tw) − epsilon × sigma × (Tw⁴ − T_rad⁴)
```

Mechanical **solves both the surface and interior temperature**, starting uniformly at 300 K. Native `CONV` and `RAD` loads evaluate convection and radiation from the current solved surface temperature during nonlinear thermal iterations. No `D,TEMP` surface constraint is applied in this mode. The equivalent convection driving temperature `B/H` is a fitted parameter, **not a sampled gas temperature or measured adiabatic recovery temperature**. `H` is spatially varying but stays constant in time; the heat flux changes with wall temperature.

This replaces the previous prescribed-temperature coupling, which remains an explicitly selectable compatibility mode. The radiation calculation in CFD identifies the equilibrium calibration anchor; radiation is also evaluated in the solid energy balance, not transferred twice as a net CFD load. See [the implementation and assumptions](docs/aerodynamic_surface_load.md).

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

The updater backs up source/configuration under `logs/source_backups/`, preserves CAD, meshes and simulation files, and selects the calibrated aerodynamic surface-load mode. It retains explicit custom numerical controls and iteration budgets, flight direction and exposure duration. Unchanged primary CFD and meshes can be reused; the new calibration and solid results are regenerated as needed. A changed primary solver configuration starts a fresh solve and archives previous solver outputs under `_history/`. Both calibration states must pass their matching numerical screens before Mechanical. Prior adiabatic exports cannot satisfy this calibration.

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

Use `--stop-after aero` to finish primary CFD exports without calibration or Mechanical. Use `--stop-after calibration` to prepare and verify the matched surface-load package. Batch execution and offline reporting are:

```powershell
.\.venv\Scripts\python.exe study.py batch --batch-config batch.yaml
.\.venv\Scripts\python.exe study.py report --runs-dir runs
```

`batch.yaml` selects whether to run the solid stage. Drag and lift are sectional forces in **N/m of span** from the single radiating-wall solution. `Cd` and `Cl` require a justified common `aero.reference_area_per_unit_span_m`; a whole-vehicle or axisymmetric reference is not inferred.

The common reference width is still undecided, so the template leaves it null
and compares drag in N/m. Aerodynamic exports now separate total, pressure and
viscous drag, and `solver_force_history.csv` records total drag/lift versus
measured Fluent iterations. The default acceptance screen checks the recent
five sampled drag values against a 1% tolerance in addition to the residual
screen. Missing histories from older saved cases require a continuation, not
a fresh initialization. See [force outputs, settings and continuation commands](docs/aerodynamic_forces.md).

## Mechanical settings and physical time

```yaml
loading:
  mode: aerodynamic_surface_load
initial_temperature_k: 300.0
include_radiation: true
# Radiation parameters are taken from the primary CFD calibration.
transient:
  duration_s: 200.0
```

Mechanical APDL reconstructs the solid boundary from actual exported CFD vertices. `H` and the equivalent driving temperature are linearly interpolated along each reconstructed boundary line. Native surface convection points into the solid when the driving temperature exceeds the solved surface temperature; native radiation takes its emissivity and external radiation temperature from the CFD equilibrium calibration. Optional Mechanical overrides must match those values; inconsistent overrides are rejected rather than silently changing the equilibrium anchor.

Tungsten conductivity, density and heat capacity belong to the solid stage. The supplied constant values—`k = 170 W/(m K)`, `rho = 19250 kg/m³`, `Cp = 130 J/(kg K)`—are configurable approximations requiring verification over the studied temperature range. Emissivity is also constant, currently 0.9. This change does not introduce temperature-dependent material data, oxidation, phase change or a flight-condition sweep. Duration defaults to **200 seconds**; set `transient.duration_s: 400.0` in the selected thermal YAML to use 400 seconds. Changing duration does not create a changing flight envelope.

For separate 200/400-second configurations, copy `thermal_analysis.yaml`, change only `transient.duration_s`, then select it with `--thermal-config`. Results for one case occupy the same `thermal_analysis/` directory; resume archives invalidated prior thermal outputs under `_history/` rather than mixing durations. Reports separate different exposure settings.

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

Resume reuses valid saved primary CFD, temperature and aerodynamic exports, and the cold calibration where current, then reruns the unfinished Mechanical stage. Keep resume enabled to retain this reuse. The thermal stage reports startup, geometry, meshing, solve and result-reading progress, and streams the native meshing and solve output. Its inactivity check also watches actual MAPDL output and result-file changes; a long native command can remain active without Python stdout. The separate hard deadline still applies. An inactivity timeout by itself does not identify whether MAPDL was busy, waiting for a license or stalled.

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

`solver_temperature_history.csv` is an optional **CFD convergence diagnostic indexed by Fluent iteration**. It is not a transient solid temperature history and is not a Mechanical load. Mechanical's wall and whole-solid histories use **physical seconds**, and their temperatures are solved values. Whole-solid min/mean/max statistics include surface nodes; the means are arithmetic nodal/sample means, not volume-weighted temperatures.

## Saved outputs and plots

```text
runs/<nose>/
  _batch/config.yaml
  02_mesh.msh.h5
  04_final.cas.h5 / 04_final.dat.h5
  summary.json
  temperature/
    wall_temperatures.csv         # wall-face and separate fluid-side temperatures
    wall_temperature_nodes.csv   # ordered actual vertices and equilibrium temperature
    temperature_summary.json
    fluid_temperature_nodes.csv  # optional supported fluid-node samples
  aero_summary.json
  cfd/cold_calibration/          # auxiliary cold-wall case/data and signed flux export
  thermal_bc/
    surface_load_faces.csv      # matched RE/cold calibration and affine face law
    surface_load_nodes.csv      # projected Mechanical calibration fields
    surface_load_summary.json   # assumptions, diagnostics and all source hashes
  thermal_analysis/
    wall_temperature_history.csv   # solved wall temperatures versus physical time
    temperature_history.csv        # all-solid nodal min/mean/max versus physical time
    surface_heat_flux_history.csv  # convection in, radiation out, net in; W/m²
    surface_energy_history.csv     # time-integrated local fluxes; J/m²
    global_energy_history.csv      # boundary power W/m and energy J/m of span
    thermal_snapshots.npz          # selected solid fields, including initial state
  reports/                       # equilibrium wall/gas profiles and maps
  solver_diagnostics/
  batch_logs/
  _history/
runs/comparison/
  radiative_equilibrium_temperature_summary.csv
  radiative_equilibrium_report.json
  solid/                         # solved wall/solid, surface-load and energy comparisons
```

Each nose gets normalized back → nose → back temperature profiles, wall and gas sample maps, optional CFD convergence curves, and separate Mechanical temperature fields and physical-time histories. Enable `visualization.make_gif` for an animation. Comparisons include equilibrium surface temperature versus sectional drag, solved transient wall/solid temperatures, local heat flux and accumulated energy, and global boundary power/energy. Local flux/energy comparison curves explicitly use simple means over saved wall samples; use global histories for spatially integrated heat input. Surface energy is calculated from the saved temperature-dependent load law using trapezoidal time integration; it is not a native internal-energy conservation certificate.

Heat flux and accumulated heat load also have maximum/mean histories, final normalized surface profiles and boundary maps, separately showing convection, radiation and net heating. The maximum-temperature bar chart uses each nose's peak over all saved solid nodes and physical times, rather than only its final temperature. Run `study.py report --runs-dir runs` to regenerate these figures without Ansys; see [plot filenames and definitions](docs/aerodynamic_surface_load.md).

The current thermal template excludes the flat rear closure using `calibration.exclude_rear_face: true`. It keeps the solid closed but treats the rear as an insulated cut: no applied gas heating or radiation, and no modeled attachment conduction. Each nose's flux/load plots and energy integrals cover the included nose/top/bottom surfaces. Full-solid temperatures and maximum-temperature bars still include the entire solid. CFD geometry, wake and full-body drag/lift integration retain the rear. Updating rebuilds the affected thermal results while reusing compatible saved CFD states.

Current reports verify mode, both calibration case/data pairs, boundary-package provenance and solid result hashes. Incompatible flight, gas, wall-radiation, mesh, material or exposure settings are excluded or separated. Existing steady RE plots retain equilibrium labels; new Mechanical surface histories are labeled as solved. Historical prescribed-surface/adiabatic plots are not relabeled as new solved-temperature results.

Historical supplied CSVs remain available explicitly:

```powershell
.\.venv\Scripts\python.exe study.py report --reference-dir examples/reference_results --output-dir runs/historical_comparison
```

Historical data are reference fixtures, not newly validated calibrated transient results. The earlier prescribed-temperature, cold/adiabatic endpoint-convection and direct-adiabatic/specified-`h` implementations remain compatibility paths for explicitly selected models. The active calibration uses **RE plus cold**, not a cold/adiabatic pair.

## Validation limits

Offline tests cover wall-condition adapters, matched-state calibration and signs, finite-heating/energy calculations, native surface-load commands, numerical screening, source/result provenance, configuration migration, comparisons and Git updates:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The cloud environment has client libraries and offline plotting, but no licensed Fluent or Mechanical engines. Windows execution is still required to verify both calibration solves and native nonlinear FE execution. The two-anchor affine heating relation is an approximation, not a universally valid high-temperature heat-transfer correlation. It extrapolates outside the calibration interval without clipping. Projection from CFD faces to FE boundary fields introduces documented interpolation errors; even imported `T_eq` need not be exactly stationary at every projected node. For CDR, assess temperatures/forces, energy balance, mesh/timestep/numerical sensitivity, high-temperature gas/tungsten properties and the planar approximation. Successful exports or a residual screen alone do not establish survivability or an optimum nose.

A fresh setup uses Python 3.12 or later and `pip install -r requirements.txt`. Cloud setup uses `bash scripts/setup_cloud.sh`, then `source scripts/activate_cloud.sh`.
