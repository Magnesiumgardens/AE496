# Internal heating comparison at 10 mm

`study.py internal-compare` solves a separate transient solid model for each
selected nose using its existing, verified Mach-8 heating calibration. It
does not run Fluent or recalculate drag. It writes to `internal_analysis`,
so the existing tungsten `thermal_analysis` results remain available.

All noses in a comparison use the same material, initial temperature,
exposure duration, depth convention, material orientation and screening
limit. Different flight, wall-radiation or rear-boundary assumptions are
excluded and explained in `internal_comparison.json`.

## Choose what the measurement means

The uploaded reference project does not contain its Mechanical mesh, probes
or result database. Consequently two physical choices in
`internal_screening.yaml` intentionally require a value:

| Setting | Meaning |
| --- | --- |
| `depth_mode: axial_tip` | One point 10 mm behind the forwardmost tip along the verified calibration's nose direction (derived from `reporting.nose_direction`). This is an axial distance, not necessarily 10 mm from the nearest heated wall. |
| `depth_mode: surface_normal` | Points 10 mm inward from every exposed boundary-segment midpoint. The comparison uses the highest temperature across accepted points. |
| `material_axes_2d: [x, y]` | Material fabric directions X/Y mapped to the model's global X/Y. |
| `material_axes_2d: [x, z]` | Material X fabric direction along global X, and material Z through-thickness along global Y. Use only if that is the intended layup. |

The material axes are fixed global directions. They do not rotate along a
curved skin. Neither probe option invents a 10 mm hollow wall or an inner
surface: both sample the existing **filled planar solid** reconstructed from
the CFD nose boundary. A hollow nose would require its actual solid geometry
and an inner-surface boundary condition.

The default `require_min_surface_clearance: true` excludes surface-normal
points that are closer than 10 mm to another heated face. Thin noses may
have no eligible points. Their results are excluded rather than moving the
probe to another depth. Setting this to `false` retains normal offsets that
are inside the solid, but their recorded nearest-wall clearance may be less
than 10 mm. Every accepted/excluded point and its reason is recorded.

Surface-normal sampling follows CFD polygon-segment midpoints. Its maximum
is a sampled maximum, not a mathematical maximum over a continuous depth
contour. Check spatial sampling and solid mesh refinement before relying
on a small difference between designs. Probe temperatures use six-node FE
shape functions at the requested coordinates, not nearest-node temperatures.

## Material and surface heating

The default material file contains the supplied Engineering Data XML's
**orthotropic carbon–carbon**, not tungsten: density 1600 kg/m³; temperature-
dependent conductivity and specific heat; material X/Y in the fabric plane,
Z through-thickness. Temperatures are converted from Celsius to Kelvin.
The supplied material table has mixed-confidence provenance and spans
255.556–2200 K. See the [reference audit](internal_heating_reference_audit.md).

The screening YAML explicitly selects `material_extrapolation: hold_endpoint`:
outside that range, property values stay at the nearest table endpoint;
temperature itself is never clipped. Results, CSVs and comparison charts
flag use of this approximation. Select `error` to reject any analysis with
nodal temperatures outside the supplied range. No oxidation, ablation,
material degradation or emissivity curve was supplied or added.

Each nose retains its existing spatial convection law and radiation constants:

```text
q_conv(s,Tw) = H(s) × [T_driving(s) − Tw]
q_rad(s,Tw)  = epsilon × sigma × [Tw^4 − T_rad^4]
q_net       = q_conv − q_rad
```

The verified cold-wall and radiative-equilibrium CFD states identify `H`
and the effective driving temperature. Cold-wall flux is an anchor, not a
constant transient heat load. Changing solid conductivity does not change
these gas-side anchors; changing the assumed surface emissivity would
require a consistent new calibration. The current calculation inherits the
calibration's constant emissivity, even though the new material is C/C.

MAPDL solves full transient 2-D conduction from uniform 300 K, with native
temperature-dependent `KXX`, `KYY` and `C`, convection on the exposed solid,
and SURF151 radiation to the calibrated surroundings. Solid surface
temperatures are solved, not prescribed. The calibrated rear cut remains
insulated when that is the existing case setting; conduction into an attached
vehicle is not modeled. Out-of-plane conduction is absent.

This remains a one-way coupled planar, laminar, fixed-flight Mach-8 model.
The affine gas-heating approximation, calorically perfect air assumptions,
CFD spatial projection, constant emissivity and missing transition/chemistry
limit quantitative accuracy. There is no new trajectory, Mach or AoA sweep.

Energy verification integrates the **exact local specific-enthalpy function**
`integral(cp(T) dT)` using FE quadrature, and compares stored solid energy
with independently integrated convection minus radiation at every native
backward-Euler substep. Spatial integration across property-table knots is
numerical. The existing conservation tolerance is retained. A failed check
excludes the result; inspect `energy_balance.json` and refine time/mesh
resolution rather than treating the numerical screen as validation.
The new internal path starts at 0.001 s, caps native steps at **0.05 s for
the first 5 s**, then allows steps up to 0.5 s. This targeted startup
refinement responds to the first native MK3 runs: final energy differed by
about 0.032%, but the worst individual step near 0.81 s exceeded tolerance.
It is a refinement to test, not demonstrated time independence. The physical
loads and temperature state continue across the two native load steps; there
is no second initialization or change to Mach number. Result sets are read
by global index across both load steps.
`startup_duration_s` and `startup_max_time_step_s` control the early phase;
setting startup duration to zero disables it. Existing internal YAML files
that omit these fields use the 5 s / 0.05 s defaults, recorded in the receipt.
The older thermal workflow keeps its existing single-phase behavior.
If the enthalpy check fails near startup, reduce `initial_time_step_s`;
also reduce `startup_max_time_step_s` when the failing time lies in startup.
For later failures reduce `max_time_step_s`. Inspect the recorded worst
step and energy terms before drawing physical conclusions.

Messages saying that loads on unselected lines were not transferred at
`SOLVE` refer to the deliberately blocked **second** solid-model transfer.
Convection was already transferred to the solid before the radiation
elements were created. These messages do not establish that convection is
missing. Do not delete the line loads to silence them: MAPDL's `SFLDELE`
also deletes their corresponding finite-element loads.

## Native execution and result files

The internal-analysis runner now explicitly launches **shared-memory MAPDL**
(`-smp`). `mapdl_parallel_mode: smp` is the default even in an existing YAML
that omits this key. `processors: 4` therefore requests four threads in one
process. An explicit `mapdl_parallel_mode: dmp` retains the distributed option.

This execution change responds to the reported Windows 2025 R1 `SIG$ABORT`
after the two-load-step solve reached 200 s: fresh distributed `.rth` parts
were present, while the combined `.rth` still belonged to an earlier run.
That demonstrates that the old combined file cannot validate the new run;
it does not identify the exact internal MAPDL defect or prove an overall
memory-exhaustion cause. Shared-memory execution avoids the multi-process
result-merging path and needs verification on the licensed host.

The internal path stores **NSOL at every solved substep**, instead of every
element result as well. For this thermal-only model NSOL contains nodal
temperatures. All nodes and physical output times needed by the probe,
convection/radiation and FE-enthalpy calculations are retained. Those
calculations already use solved temperatures and the verified mesh/load laws;
they do not require saved native element heat-flux or reaction records.
The conduction equations, load laws and energy gate are unchanged. The
separate older thermal workflow retains its existing native output policy.

Each new attempt has its own directory:

```text
runs/<case>/internal_analysis/native_runs/<UTC attempt ID>/
    internal_model.db       # model saved before the first SOLVE
    internal_nose.rth       # thermal results, when written by MAPDL
    mapdl_commands.log
    mapdl_console.out
```

The path is printed before launch and recorded under `native_runtime` in
`internal_summary.json`, including on failure. The original native files in
the parent directory are retained. A fresh attempt cannot accidentally read
those older combined results. Neither a partial distributed result nor a
final convergence message bypasses the result-reading and energy checks.

For an initial controlled retry of the failed nose, use one shared-memory
thread without editing the exposure/material YAML:

```powershell
.\.venv\Scripts\python.exe study.py update
.\.venv\Scripts\python.exe -u study.py internal-compare --screening-config .\internal_screening_carbon_carbon.yaml --cases "mk3_body_internals_blunt_nose" --processors 1
```

`--processors` changes runtime execution only; it does not change the physical
comparison settings or force already-valid results to be recalculated when
using `--report-only`. Existing results still need their original verified
material, calibration, physical settings and source hashes.

## Run on Windows

The filled example `internal_screening_carbon_carbon.yaml` uses the supplied
C/C composite properties, a **10 mm axial offset behind the tip**, material
axes **[x, z]** mapped to global X/Y, 300 K initial temperature, 200 s exposure
and the slide's 1873.15 K limit. These are explicit example measurement/layup
assumptions, not recovered probe definitions. The table represents the
effective carbon–carbon composite, not its matrix constituent alone.

After updating, run that example with:

```powershell
.\.venv\Scripts\python.exe -u study.py internal-compare --screening-config .\internal_screening_carbon_carbon.yaml --cases "*mk3*" "*mk4*"
```

The generic configuration below leaves the measurement and layup unselected.
From the AE496 folder, update first:

```powershell
.\.venv\Scripts\python.exe study.py update
notepad .\internal_screening.yaml
```

Set `depth_mode` and `material_axes_2d` according to the actual measurement
and material layup above. Keep `depth_m: 0.010`,
`limit_temperature_k: 1800.0`, and `duration_s: 200.0` for the requested
screen. The later MK3 slide specifies **1873.15 K**; set the limit to that
value to use its temperature criterion. Its 143 s reference follows Mach 8
until 85 s, then Mach 7 until 170 s. This runner remains at Mach 8 throughout;
it does not reproduce that flight history. The slide's 1347.15 K line is a
provisional temperature-rise margin, not a structural factor of safety.
Change duration to 400 to run that exposure instead; the same
`internal_analysis` destination is overwritten, so copy results/configuration
first if retaining both durations.

Run the MK3/MK4 saved cases selected from `geometry_batch`:

```powershell
.\.venv\Scripts\python.exe -u study.py internal-compare --cases "*mk3*" "*mk4*"
```

Omit `--cases` to evaluate every CAD case. Files must already have valid
matched cold/RE calibration in their case folder; a changed or missing
calibration is reported and requires the existing CFD/calibration workflow.
The internal runner does not silently rerun Fluent or substitute another nose.

When a case already has a valid saved primary RE solution but lacks
`thermal_bc/surface_load_summary.json`, prepare that case's calibration
directly, without running the aerodynamic export:

```powershell
.\.venv\Scripts\python.exe -u run_calibration.py --config .\runs\mk4_body_internals_blunt_nose_v2\_batch\config.yaml
```

This reuses valid cold-wall exports when available; otherwise it runs the
required cold-wall calibration on that case's mesh. It requires the saved
primary CFD case/data to exist and pass their existing checks.

To regenerate plots from saved internal results, without a new MAPDL solve:

```powershell
.\.venv\Scripts\python.exe study.py internal-compare --cases "*mk3*" "*mk4*" --report-only
```

Report-only verifies the selected CAD, calibration, material, configuration
and saved results. Changing these inputs requires a new solid analysis.
Updates install the new reference material and preserve a screening YAML
that already exists locally.

If the two MK3 models were originally given each other's names, correct the
comparison using saved results:

```powershell
.\.venv\Scripts\python.exe study.py internal-compare --screening-config .\internal_screening_carbon_carbon.yaml --cases "*mk3*" "*mk4*" --report-only --swap-labels mk3_body_internals_base_nose1 mk3_body_internals_blunt_nose
```

This swaps display names on the internal-temperature and surface-load
comparisons. Comparison CSVs retain the original `source_case` alongside
the corrected `design` label; the comparison receipt records the mapping.
It does not rename CAD, run folders or per-case data, or alter source hashes.
Include the same flag whenever regenerating these corrected comparisons.
Omit it to return to the original labels. Both names must be in the selection.

### One command for a 170-second carbon–carbon transient

The convenience entry point includes the confirmed MK3 name swap, one
shared-memory thread, and the existing C/C example configuration:

```powershell
.\.venv\Scripts\python.exe run_internal_170.py
```

Use `run_internal_170.py --report-only` to regenerate its saved 170 s reports.
This wrapper selects the MK3/MK4 cases. The equivalent general command is below.

Use `--duration-s 170` to solve a physical exposure ending at exactly 170 s
without editing the example YAML. This runs the internal C/C solid models
using their already-prepared CFD calibrations, with the corrected MK3 names:

```powershell
.\.venv\Scripts\python.exe -u study.py internal-compare --screening-config .\internal_screening_carbon_carbon.yaml --duration-s 170 --processors 1 --cases "*mk3*" "*mk4*" --swap-labels mk3_body_internals_base_nose1 mk3_body_internals_blunt_nose
```

This is a new 170 s solve, not a relabeling of a saved 200 s result. It
overwrites the selected cases' derived internal-analysis results; native
attempt directories remain separate. The duration is recorded in the physics
settings. Missing calibration still excludes that case rather than silently
running a different workflow or substituting another nose's loads.

Each case gets temperature histories, its final solid field, surface heat
flux/heat-load CSVs and derived figures under `internal_analysis/plots/`.
The combined flux, heat-load histories, final spatial distributions at
170 s and bar charts are in `comparison/internal_heating/surface_loads/`.
Convection into the solid, radiation out and net flux remain separate. The
peak-internal-temperature bar now explicitly states the 0–170 s interval.

To regenerate those reports later without solving, repeat the same command
with `--report-only` added, retaining `--duration-s 170` and the label mapping.
Saved 200 s results do not satisfy a request for a newly solved 170 s exposure.

## Outputs and interpretation

Each `runs/<nose>/internal_analysis/` contains:

- `internal_temperature_history.csv` / `.png`: each probe versus physical
  seconds, including the initial state; coordinates and depth definition.
- `internal_worst_temperature_history.csv`: sampled maximum across probes.
- `internal_probe_screening.csv`: first crossing and saved-time bracket for
  each probe. The overall crossing is the earliest individual crossing.
- `internal_temperature_final.png`: final solid field with probe positions.
- `internal_solution.npz`: all solved nodal fields and physical times, with
  six-node triangle connectivity.
- The surface-temperature, heat-flux, heat-load and solid-energy CSVs also
  written by the existing thermal result writer.
- `internal_summary.json`: complete/failed status, source hashes, material
  and orientation, probes, screening result, extrapolation, energy check and
  native execution directory/settings.

`runs/comparison/internal_heating/` contains the internal-temperature
history overlay, peak internal-temperature bar chart and time-to-1800-K bar
chart, their CSVs, and the comparison receipt. A hatched `>200 s` bar means
the limit was not reached within the simulation; it is not an extrapolated
failure time. The 143 s reference is an annotation, never a fitted target,
and appears only when the selected limit matches its 1873.15 K criterion.

This is a **thermal screening criterion**, not a structural failure model.
A lower peak heat flux alone does not establish a proportional reduction in
internal temperature or a proportional increase in time to the limit.
Spatial loading, geometry, conduction directions, heat capacity, radiation
and exposure all affect that result. It cannot establish that another
vehicle component becomes the limiting factor.

Offline tests cover tables, FE interpolation, enthalpy, thresholds, native
command construction and report/source checks. Native MAPDL execution and
mesh/time convergence must be checked on the licensed Windows host.
