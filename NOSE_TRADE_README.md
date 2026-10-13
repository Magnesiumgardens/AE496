# Nose trade study update

This is an **overlay for your existing Windows project**, not a complete project
or an Ansys installation. It contains the supplied files plus the aerodynamic
and thermal screening changes. Keep your existing `pipeline/config.py`,
`geometry.py`, `mesh.py`, `thermal_export.py`, `utils.py`, `run_all.py` and other
entry points. No missing module has been replaced with a placeholder.

Back up the files being replaced, then copy this overlay into
`hypersonic_thermal_pipeline`, preserving the `pipeline/` and `tests/` paths.
Review the YAML against any local changes before replacing it. No commits or
pushes have been made to GitHub.

## What is included

- `pipeline/aero.py`: total drag and lift projected along/perpendicular to the
  configured freestream, integrated over `body_wall` and its split wall zones.
  Reference depth is explicitly one metre and outputs are **N/m of span**.
  `Cd`, `Cl`, and signed `L/D` are included when their normalization is defined.
  Zero drag produces an undefined `L/D`, not infinity. Farfield forces are excluded.
- `pipeline/solve.py`: force reporting after iterations, saved under
  `aerodynamics` in the existing CFD summary and in `aero_summary.json`; missing
  meshes fail before Fluent starts. Force report failures are recorded explicitly
  without discarding CFD/thermal export work.
- `export_aero_only.py`: read existing final case/data and compute forces without
  meshing, initialization, iterations, or rewriting the original case/data.
  Flight conditions, wall thermal conditions, and material/viscous model types
  are checked against the saved case before reporting. Convergence is not certified.
- `compare_nose_trade.py`: compare existing force and thermal results without
  launching Ansys or rerunning the batch. Saved temperature histories supply
  peak temperature; missing heating, mass or limit metrics are not invented.
- `run_batch.py`: backfill missing/stale force reports, extend CSV/JSON output,
  retain existing thermal plots, add force/coefficient trade plots, and emit
  `nose_trade_screening.json` with screening Pareto candidates. Different flow,
  reference, thermal/material, or boundary-layer settings are not ranked together.
  Generic launch/API/license failures no longer trigger inflation-off retries.
  Unknown boundary-layer log markers conservatively produce no retry.
- `run_thermal_analysis.py`: for explicit fixed-heat-flux screening, add peak
  transient temperature, time above a configured temperature limit, margin to
  that limit, applied heat load, and mass per span of the filled thermal polygon.

## Important thermal-model mismatch in the uploaded files

The supplied `thermal_analysis.yaml` requests `convective_from_cfd`, but the
supplied Python runner applies `HFLUX` and never reads the adiabatic CSV. Its
original summary even states that flux remains fixed as the wall heats up.

This update **does not silently downgrade that request**. The YAML retains
`convective_from_cfd`, and the thermal runner and batch preflight reject this
unsupported mode before launching expensive jobs. To run your intended model,
provide the endpoint-convection thermal runner and the missing
`run_adiabatic_reference.py`, including its CSV format. Those sources were not
uploaded. Existing endpoint thermal results are not stamped as fixed-q results.

For an explicitly chosen **legacy fixed-q screening** study, set
`loading.mode: fixed_heat_flux` in `thermal_analysis.yaml` and
`run_adiabatic_reference: false`, `adiabatic_required: false` in `batch.yaml`.
This is a different thermal assumption, not a repair of endpoint convection.
Do not present these temperatures as the intended temperature-dependent wall model.

## Configuration

The `aero` block in `config.yaml` is enabled for confirmed **planar sections**.
It deliberately sets `reference_area_per_unit_span_m` and `reference_length_m`
to null. Choose one justified reference width (`A_ref / span`, units m) shared by
all geometries to obtain `Cd = drag_per_span / (q_inf * reference_width)`.
No frontal area is guessed. A reference length is optional for the present
force calculation; moments are not yet implemented.

Normalization uses calorically perfect air with explicit gamma = 1.4 and
R = 287.05 J/(kg K): rho = p/(RT), V = M sqrt(gamma RT), q = gamma p M^2 / 2.
Check that these assumptions suit your Fluent air model. At Mach 8, high-temperature
gas effects may matter; this update does not certify perfect-gas adequacy.

`design_limits.max_temperature_k` in `thermal_analysis.yaml` remains null.
Enter a justified operating limit for your actual material/TPS, rather than
assuming its melting point is an acceptable service temperature. Without a
limit, feasibility is explicitly unassessed. The time-above-limit metric uses
piecewise-linear saved maximum-temperature history, including the initial state.

The configured SpaceClaim launch mode reflects your successful diagnostic;
retain the explicit SpaceClaim launcher fix in your local `pipeline/geometry.py`.
That file is not part of this overlay. Existing generated case configurations
may lack the new aero block; update their aero settings when choosing a width.
The batch runner generates new case configs from the base config.

## Test one case first

From the Windows project root, keep using the explicit virtual-environment Python.
For an existing solved `blunt_1` case:

```powershell
.\.venv\Scripts\python.exe export_aero_only.py --config ".\runs\blunt_1\_batch\config.yaml"
Get-Content ".\runs\blunt_1\aero_summary.json"
```

This requires both `04_final.cas.h5` and `04_final.dat.h5`. If only the mesh exists,
run your corrected solver-only entry point first:

```powershell
.\.venv\Scripts\python.exe run_solver_only.py --config ".\runs\blunt_1\_batch\config.yaml"
```

Force reports use the PyFluent 0.42.1 generated Fluent 251 settings interface.
The exporter checks available output enums and fails clearly rather than
interpreting an unknown default coefficient as a force. A report error is not
a zero drag result; failed reports do not participate in ranking.

Resolve the thermal-runner mismatch before starting a complete batch:

```powershell
.\.venv\Scripts\python.exe run_batch.py --batch-config batch.yaml
```

You can compare existing completed thermal runs immediately after exporting
their forces, without changing their thermal assumptions:

```powershell
.\.venv\Scripts\python.exe compare_nose_trade.py --runs-dir runs
```

This writes `runs/nose_trade_comparison`. Export forces for every candidate first.

After an eligible batch, comparison output adds sectional drag by geometry,
peak temperature versus drag, and peak heat flux versus drag. Equivalent Cd
plots appear only when coefficients have an explicit reference width. Starred
temperature/drag points are screening Pareto candidates, not an automatic winner.

## Interpretation and remaining work

These are screening results. Validate residuals, force/heat monitor stability,
mass balance, boundary-layer resolution and mesh sensitivity before deciding.
Do not mix inflation-on and inflation-off cases to claim a fair drag/heating trade.
The iteration count alone does not prove convergence.

Applied thermal heat load is the signed prescribed fixed-q input before
radiation loss; it is not actual time-varying convective energy. Area and mass
are estimates from wall-face-centroid polygon reconstruction, assuming a filled
uniform-material section. They do not represent TPS-shell mass or exact CAD.

Pressure/viscous drag decomposition, pitching moment, center of pressure,
angle-of-attack stability sweeps, thermal stress, ablation, variable material
properties and mission envelopes remain further study work. The present update
does not infer a 3-D vehicle or select arbitrary weights to declare a best nose.

## Validation performed in the cloud

The unit tests exercise reference normalization, rotated force directions,
body-wall scope, ambiguous report results, stale outputs, saved-case mismatches,
thermal exposure integration, CSV/JSON propagation, retry classification and
Pareto plotting with synthetic fixtures. They do not run a licensed Fluent or
MAPDL session. The cloud checkout lacks the rest of the source and your Windows
Ansys installation. Do not call the complete application validated from these tests.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The existing project needs PyFluent plus numpy, PyYAML and matplotlib; the
thermal workflow additionally needs PyMAPDL and licensed Ansys backends. No new
third-party runtime package is introduced by this overlay.
