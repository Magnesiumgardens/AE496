# AE496 — hypersonic nose study for CDR

A planar 2-D Ansys 2025 R1 workflow: **mesh → cold-wall CFD → warm-started adiabatic CFD → spatial convection inputs and aerodynamic forces → Mechanical APDL transient tungsten analysis → individual and comparison plots**.

## What the two CFD solutions provide

The restored default follows the method declared in your uploaded original configuration:

1. Solve the steady flow with the nose wall prescribed at 300 K. Export the local cold-wall heat flux.
2. Read that converged case/data into a second steady solve, change the wall to **Heat Flux = 0 W/m²**, and continue without reinitializing. Export the adiabatic wall/recovery temperature, `T_aw`.
3. On matching wall faces, derive `h(s) = q_cold_into_solid(s) / (T_aw(s) − T_cold)` and map the face values to the shared boundary nodes for Mechanical.
4. Solve tungsten conduction in Mechanical using spatial convection inputs `h(s)` and `T_aw(s)`, plus configured radiation. Fluent does not perform the solid transient analysis.

The cold-wall flux is a calibration measurement. It is not held as a fixed heat-flux load throughout solid heating. Mechanical calculates the changing convection load as `h(s) [T_aw(s) − T_solid(s,t)]`.

```yaml
workflow:
  strategy: two_endpoint
thermal:
  mode: fixed_temperature
  wall_temperature_k: 300.0
```

The root `thermal` setting describes the cold-wall endpoint. The generated adiabatic endpoint overrides it internally. Flow, geometry, gas properties and mesh are shared; the nose sees airflow toward **−X**. Runtime checks verify that both saved endpoints satisfy their intended wall conditions and residual screens before deriving convection inputs. Heat-flux sign is explicit rather than inferred from an absolute value. Invalid temperature differences, incompatible samples or negative coefficients fail instead of silently producing a load.

`h(s)` is an effective coefficient evaluated at the cold-wall reference temperature. It is held fixed through the configured solid exposure. This release does not add an `h(T_wall, flight)` lookup table, a time-varying flight envelope or two-way coupling. Those require additional calibration data or justified correlations.

## Update and run on Windows

The maintained branch is `codex/nose-trade-foundation`. Existing installations update through authenticated Git:

```powershell
.\.venv\Scripts\python.exe study.py update
.\.venv\Scripts\python.exe study.py run --config ".\runs\blunt_1\_batch\config.yaml" --stop-after thermal
```

The updater backs up code and configuration under `logs/source_backups/`, preserves CAD and simulation files, migrates case settings to the endpoint workflow, and retains the unchanged mesh. Previous direct-adiabatic results are retained as history; they cannot substitute for a verified cold-wall endpoint. Material and exposure choices are preserved where supported by the migrated loading model.

If the current installation predates `study.py update`, use a temporary Git checkout:

```powershell
$src = Join-Path $env:TEMP ("AE496-source-" + [guid]::NewGuid().ToString("N"))
git clone --single-branch --branch codex/nose-trade-foundation https://github.com/Magnesiumgardens/AE496.git $src
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe "$src\scripts\update_project.py" --git-repo "$src" --root "."
}
```

For a case configuration outside `runs/*/_batch`, use `study.py migrate --config "path\to\case.yaml"`. Updates use an owned `.runtime/source_checkout` and Git `--ff-only`; the updater does not reset your project checkout.

For a new nose, prepare and inspect the mesh before solving:

```powershell
.\.venv\Scripts\python.exe study.py run --geometry ".\geometry_batch\blunt_1.x_t" --stop-after mesh
.\.venv\Scripts\python.exe study.py run --geometry ".\geometry_batch\blunt_1.x_t" --stop-after thermal
```

Geometry must be one XY fluid surface: outer farfield minus the body hole. SpaceClaim 251 is selected for this installation. The Fluent 2-D meshing workflow uses a **3-D meshing host** with Prime 2-D enabled; its output mesh and CFD solve are planar 2-D. Inspect shock resolution and the near-wall mesh. No automatic retry removes inflation layers.

To finish the CFD/calibration/aerodynamic stages without starting Mechanical, use `--stop-after aero`. To run batches or regenerate reports without starting Ansys:

```powershell
.\.venv\Scripts\python.exe study.py batch --batch-config batch.yaml
.\.venv\Scripts\python.exe study.py report --runs-dir runs
```

Set `run_thermal: true` and `thermal_config: ./thermal_analysis.yaml` in `batch.yaml` for the solid stage. Drag and lift are sectional forces in **N/m of span**, exported from the adiabatic endpoint. `Cd` and `Cl` require a justified common `aero.reference_area_per_unit_span_m`; no whole-vehicle or axisymmetric area is guessed.

## Mechanical inputs and exposure

The default `thermal_analysis.yaml` uses `loading.mode: convective_from_cfd`. A manual uniform `h` is no longer required for this strategy; the derived spatial coefficient is the Mechanical input. The older direct-adiabatic strategy remains available with its explicitly specified uniform coefficient.

Mechanical APDL (MAPDL) reconstructs the planar solid boundary from exported CFD vertices and applies `CONV` loads. Tungsten conductivity, density and specific heat belong to this solid stage. The supplied constants (`k = 170 W/(m K)`, `rho = 19250 kg/m³`, `Cp = 130 J/(kg K)`) are approximations that need verification over the relevant temperature range.

`transient.duration_s` defaults to **200 seconds**. Initial solid temperature is 300 K, and the restored baseline enables surface radiation with configurable emissivity and surroundings temperature. Neither the emissivity nor constant material properties establishes a validated high-temperature tungsten model. The current calculation holds one flight condition throughout the exposure; changing duration does not create a changing flight envelope.

Changing only solid material settings or duration reuses valid CFD/calibration results and reruns Mechanical. To rerun the solid analysis directly from a current convection package:

```powershell
.\.venv\Scripts\python.exe study.py thermal --config ".\runs\blunt_1\_batch\config.yaml"
```

## Numerical controls and continuation

The cold-wall baseline uses hybrid initialization, as declared by the original solver. Courant number and flow scheme are left at Fluent's native defaults unless explicitly configured; the selected values are recorded. The adiabatic endpoint always starts from the screened cold-wall field and does not initialize again.

The default residual screening targets are **1e-3 for flow and energy**, matching the restored baseline rather than the earlier rebuild's tighter `1e-6` energy target. Fluent's native stopping criteria and the script's screen are aligned. These are acceptance thresholds, not temperature error bounds or proof of accuracy. The screen also rejects final temperature/pressure limiting and nonfinite residuals. Failed solves remain available for diagnosis and cannot supply Mechanical loading.

An unconverged but stable endpoint can be continued without discarding its progress:

```powershell
.\.venv\Scripts\python.exe study.py continue --config ".\runs\blunt_1\_batch\config.yaml" --endpoint cold_wall --iterations 4800
.\.venv\Scripts\python.exe study.py continue --config ".\runs\blunt_1\_batch\config.yaml" --endpoint adiabatic --iterations 4800
```

Use the endpoint which failed. A deliberate Courant change can be supplied with `--courant`; it should be informed by the residual and field behavior. Prior configurations and solutions are backed up. Candidate files are retained for recovery, and an optional raw-history file lock cannot roll back a valid solution. Residual diagnostics live with the corresponding endpoint. Repeated oscillation or clipping requires investigation of numerics, mesh and physics rather than an unlimited iteration budget.

`solver_temperature_history.csv`, when enabled and supported by Fluent, is an **optional convergence diagnostic indexed by iteration**. It is not a flight-time history and is not passed to Mechanical. Native measured iteration labels are preserved. Mechanical's `temperature_history.csv` and `wall_temperature_history.csv` use physical seconds.

## Files and plots

```text
runs/<nose>/
  _batch/config.yaml
  02_mesh.msh.h5
  cfd/
    cold_wall/       # cold-wall case/data, summary and numerical diagnostics
    adiabatic/       # warm-started adiabatic case/data and diagnostics
  thermal_bc/
    convection_nodes.csv
    convection_summary.json
  temperature/
    wall_temperatures.csv
    wall_temperature_nodes.csv
    temperature_summary.json
    fluid_temperature_nodes.csv   # optional supported fluid-node samples
  aero_summary.json
  thermal_analysis/   # Mechanical temperatures, time histories and figures
  reports/            # CFD recovery-temperature and derived-h plots
  batch_logs/
  _history/
runs/comparison/       # recovery temperature, sectional drag/lift comparisons
  solid/               # physical-time solid histories and final wall profiles
```

Gas exports distinguish wall/recovery temperatures from fluid-side surface values. Face coordinates are wall-face centroids, not fluid-cell centers or a specified offset from the wall. `wall_temperature_nodes.csv` records Fluent's interpolated boundary temperatures. The Mechanical package uses actual boundary vertices with adjacent-length-weighted projections of the collocated face coefficient and recovery temperature. It documents this projection; nodal interpolation need not reproduce every calibration-face heat flux exactly. These temperatures are not transient solid predictions.

Per-nose and comparison reports include temperature maps, normalized back → nose → back profiles, derived spatial `h`, drag/temperature comparisons, and separate Mechanical temperature fields and physical-time histories. Current reports check endpoint, package and result hashes; stale outputs are excluded. Comparisons separate incompatible flight, physics, mesh, material or exposure settings. Spatial `h` distributions naturally differ by nose and are part of the nose comparison.

Each nose also has a cold-wall calibration flux profile and map. These are labelled calibration data; Mechanical evaluates its changing convection and radiation load separately as the solid warms.

Historical supplied plots remain reproducible explicitly:

```powershell
.\.venv\Scripts\python.exe study.py report --reference-dir examples/reference_results --output-dir runs/historical_comparison
```

Historical CSVs are references, not newly validated endpoint results. The evidence and differences in the uploaded original are documented in [docs/ORIGINAL_WORKFLOW_COMPARISON.md](docs/ORIGINAL_WORKFLOW_COMPARISON.md).

## Validation and CDR use

Offline tests cover endpoint conditions and warm starts, sampling and heat-flux sign, spatial convection derivation, numerical screening, configuration migration, result provenance, reporting and Git updates:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The cloud environment supports development and offline plots but has no licensed Ansys engines. End-to-end Windows execution is still required. For CDR, assess stable temperatures and forces, conservation, mesh sensitivity, higher-order discretization, gas-property assumptions and the applicability of planar 2-D flow. A residual screen or successful file export does not establish material survivability or an optimum nose.

A fresh setup uses Python 3.12 or later and `pip install -r requirements.txt`. In the cloud, run `bash scripts/setup_cloud.sh` and source `scripts/activate_cloud.sh`.
