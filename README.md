# AE496 — adiabatic hypersonic nose study for CDR

A planar 2-D Ansys 2025 R1 workflow: **mesh → adiabatic CFD → gas/wall temperature export → aerodynamic forces → per-nose and comparison plots**.

## Boundary condition and interpretation

Every vehicle-wall edge uses Fluent **Heat Flux = 0 W/m²**. No wall temperature is prescribed, and radiation is disabled. The fluid energy equation remains enabled: compression, shocks and viscous effects can heat the gas even though no heat crosses the wall.

```yaml
thermal:
  mode: adiabatic
```

The wall temperature is the CFD's adiabatic/recovery temperature. It is not a transient solid temperature or a material-temperature prediction. There is no film coefficient, cold-wall heat-load mapping, MAPDL solid solve or 400-second solid exposure in this workflow.

`wall_temperatures.csv` contains two distinct requests for the same wall faces:

- `wall_temperature_K`: face values with boundary values enabled.
- `near_wall_fluid_temperature_K`: cell-based surface values with boundary values disabled, where Fluent supports that distinction.

Both are static temperatures in kelvin. CSV coordinates are **wall-face centroids**, not fluid-cell centers or positions at a specified distance from the wall. Near-wall values depend on the mesh and Fluent's surface interpolation; they can be close to wall values under an adiabatic condition. The optional fluid-node export records actual sampled node coordinates and interpolated static temperatures throughout available interior surfaces. An unavailable optional node map is reported explicitly.

The exporter verifies the saved wall boundary condition and checks that exported wall heat flux is zero within the configured numerical tolerance (default 1 W/m²). It rejects an isothermal saved case instead of relabeling its results.

## Update your existing Windows folder

The maintained branch is `codex/nose-trade-foundation`. Retrieve updates through authenticated Git. For the first complete update, use the temporary source checkout from the preceding update commands:

```powershell
git -C "$src" pull --ff-only
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe "$src\scripts\update_project.py" --git-repo "$src" --root "."
}
```

If that PowerShell window has been closed, create a new source checkout first:

```powershell
$src = Join-Path $env:TEMP ("AE496-source-" + [guid]::NewGuid().ToString("N"))
git clone --single-branch --branch codex/nose-trade-foundation https://github.com/Magnesiumgardens/AE496.git $src
```

The complete updater backs up code/configuration under `logs/source_backups/`, preserves CAD and existing simulation files, and converts the root template and `runs/*/_batch/config.yaml` to adiabatic. It preserves flight, geometry, meshing and reference-width choices. Solver/export records are invalidated while the existing mesh record remains reusable. Old CFD and solid outputs are archived under the case's `_history/` before the new CFD solve.

**Future updates require one command:**

```powershell
.\.venv\Scripts\python.exe study.py update
```

This uses Git clone/pull with `--ff-only` in an owned `.runtime/source_checkout`, then runs the same backed-up installer. It works even if your current project folder was originally downloaded as a ZIP. It does not reset your project's Git checkout.

For a case configuration outside `runs/*/_batch`, explicitly migrate it:

```powershell
.\.venv\Scripts\python.exe study.py migrate --config "path\to\case.yaml"
```

## Run the adiabatic study

Resume `blunt_1` using its authoritative case configuration:

```powershell
.\.venv\Scripts\python.exe study.py run --config ".\runs\blunt_1\_batch\config.yaml"
```

This reuses an unchanged existing mesh, reruns CFD with the new wall condition, exports temperature/drag data and creates reports. The former 300 K solution cannot establish an adiabatic result; one new CFD solve is necessary.

For a new nose:

```powershell
.\.venv\Scripts\python.exe study.py run --geometry ".\geometry_batch\blunt_1.x_t" --stop-after mesh
.\.venv\Scripts\python.exe study.py run --geometry ".\geometry_batch\blunt_1.x_t"
```

Inspect the first mesh and boundary-layer resolution. Geometry must be one XY fluid surface: outer farfield minus the body hole. SpaceClaim 251 is selected because it works on this installation. Fluent's 2-D meshing workflow uses a **3-D meshing host** with Prime 2-D enabled; the exported mesh and CFD solver remain planar 2-D.

Batch all noses, or regenerate saved reports without Ansys:

```powershell
.\.venv\Scripts\python.exe study.py batch --batch-config batch.yaml
.\.venv\Scripts\python.exe study.py report --runs-dir runs
```

Re-export an already solved adiabatic case, or inspect its available temperature fields:

```powershell
.\.venv\Scripts\python.exe study.py export --config ".\runs\blunt_1\_batch\config.yaml"
.\.venv\Scripts\python.exe study.py export --config ".\runs\blunt_1\_batch\config.yaml" --inspect-fields
```

Airflow remains toward −X into the +X nose. Drag/lift are planar sectional forces in **N/m of span**. `Cd`/`Cl` require a justified common `aero.reference_area_per_unit_span_m`; no 3-D reference area is guessed.

## Data and plots

Each nose has one directory:

```text
runs/<nose>/
  _batch/config.yaml
  _batch/stage_state.json
  02_mesh.msh.h5
  04_final.cas.h5
  04_final.dat.h5
  summary.json
  solver_temperature_history.csv  # when an actual iteration counter is available
  aero_summary.json
  temperature/
    wall_temperatures.csv
    temperature_summary.json
    fluid_temperature_nodes.csv   # when supported
  reports/                        # individual maps, profiles and diagnostics
  batch_logs/
  _history/                       # retained superseded results
runs/comparison/
  adiabatic_temperature_summary.csv
  adiabatic_temperature_profiles.csv
  adiabatic_report.json
  wall_temperature_K_comparison.png
  near_wall_fluid_temperature_K_comparison.png
  drag_vs_near_wall_temperature.png
```

Per-nose CSVs preserve coordinates, face identity, wall distance, both temperature definitions and diagnostic wall heat flux. Wall profiles are normalized back → nose → back with the nose at 0.5. Temperature means are length weighted. Reports exclude old isothermal cases and stale exports; export provenance includes hashes of the case/data pair and CSV. Comparisons require matching flow, physics, mesh settings and force normalization.

Temperature convergence histories sample minimum, length-weighted mean and maximum wall temperature every 25 measured Fluent iterations when the backend exposes a supported actual iteration counter. Individual and cross-nose history plots label the x-axis **Fluent iteration**, with maximum solid and mean dashed. If a counter is unavailable, history capture is explicitly marked unavailable and the steady solve continues; no iteration or physical-time history is fabricated.

The older temperature-versus-time plots came from solid transient heating. Steady adiabatic CFD cannot generate a physical 400-second temperature history. Historical supplied CSVs remain available under `examples/reference_results`; reproduce their earlier comparisons explicitly:

```powershell
.\.venv\Scripts\python.exe study.py report --reference-dir examples/reference_results --output-dir runs/historical_comparison
```

Historical plots are not new adiabatic CFD results. The legacy solid thermal runner refuses execution.

## Validation and CDR use

Tests cover wall settings and pinned Fluent251 schemas, temperature sampling, zero-flux rejection, configuration migration, reuse/archive behavior, saved-case guards, reporting and updater preservation. The cloud supports development and offline plots but has no licensed Ansys engines. A Windows run is still required to verify the new adiabatic solve and temperature exports.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For CDR conclusions, assess residuals and force/temperature stability, mesh sensitivity, gas-property assumptions and the applicability of planar 2-D flow. Adiabatic recovery temperatures do not establish material survivability, insulation performance or a best nose by themselves. The current air model uses ideal-gas density and Sutherland viscosity; high-temperature gas physics may materially affect absolute temperatures.

A fresh Windows setup uses Python 3.12 or later and `pip install -r requirements.txt`. In the cloud, run `bash scripts/setup_cloud.sh` and source `scripts/activate_cloud.sh` from the existing checkout. No Git worktree is needed.
