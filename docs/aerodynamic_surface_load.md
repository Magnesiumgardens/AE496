# Calibrated aerodynamic surface heating

This is the active thermal coupling for `single_solve` /
`radiative_equilibrium` primary CFD and `aerodynamic_surface_load` Mechanical
loading. The primary flow calculation is unchanged; the extra cold-wall flow
calculation is a wall-state calibration only. No flight-condition sweep is
performed.

## Heating relation and signs

At each selected, geometrically matched CFD wall face, the code uses:

\[
q''_{eq}=\epsilon\sigma(T_{eq}^4-T_{rad}^4),\quad
H=\frac{q''_{cold}-q''_{eq}}{T_{eq}-T_{cold}},\quad
B=q''_{cold}+H T_{cold}.
\]

The transient solid boundary balance is

\[
q''_{net}(s,T_w)=\underbrace{B(s)-H(s)T_w}_{q''_{conv}\text{ into solid}}
-\underbrace{\epsilon\sigma(T_w^4-T_{rad}^4)}_{q''_{rad}\text{ out}}.
\]

Positive net heat flux heats the solid. Native Fluent cold-wall flux retains
its raw value and uses the explicitly configured sign
`calibration.heat_flux_into_wall_sign` (default −1); the calibration does not
take absolute values or automatically flip signs. Nonpositive identified
conductance or insufficient separation between calibration temperatures is
rejected.

`pipeline/aerothermal_calibration.py` exports and matches face centroids,
closed-loop endpoints, zone names, geometry and the common mesh hash. Native
face indices may differ after parallel repartitioning. The root RE solution
and the cold state each require a matching passed CFD residual screen.

The RE convective anchor is calculated from external radiation balance.
The code does not relabel Fluent's generic Radiation-wall `heat-flux` field
as separated convective flux. The two anchors identify one affine heating
law; they do not establish a nonlinear gas correlation valid for every wall
temperature. The fitted `B/H` is an equivalent driving temperature, not an
independently measured gas temperature.

## Mechanical implementation

`pipeline/aerothermal_loading.py` supplies native MAPDL `CONV` loads on the
`PLANE293` solid edges; `pipeline/thermal_radiation.py` attaches dedicated
`SURF151` radiation elements to the same exposed quadratic edges. `H` and `B/H` are each linearly
interpolated along each boundary segment; their product need not be a linear
interpolation of `B`. Native nonlinear iterations evaluate both loads using
the current FE surface temperature. No solid node receives a `D,TEMP` condition.
Only the separate ambient node is held at the surroundings temperature; it has
no solid heat capacity and is excluded from solid statistics and result files.
Cold-wall flux is not held fixed throughout exposure. Line loads are transferred
before creating radiation surfaces, then deselected before `SOLVE` to prevent
automatic convection transfer onto the radiation elements.

The initial temperature is uniform 300 K by default. Full transient
conduction uses PLANE293 thermal elements, automatic time stepping and the
configured duration. In `thermal_analysis.yaml`:

```yaml
initial_temperature_k: 300.0
loading:
  mode: aerodynamic_surface_load
include_radiation: true
# Emissivity and radiation temperature are inherited from CFD.
transient:
  duration_s: 200.0
  initial_time_step_s: 2.0
  min_time_step_s: 0.2
  max_time_step_s: 10.0
```

Set only `duration_s: 400.0` for the same conditions over 400 s. Optional
Mechanical emissivity and radiation-temperature overrides must match the
primary CFD equilibrium calibration; otherwise they are inherited from CFD.
The RE radiation term identifies an incoming gas-heating anchor; the same
physical outgoing radiation is then evaluated in the transient solid
balance. No RE net wall load or constrained temperature is added to that
balance.

## Files and units

The primary RE solution remains `04_final.cas.h5` / `04_final.dat.h5`.
The calibration solution is under `cfd/cold_calibration/`. The matched package
is in `thermal_bc/surface_load_faces.csv`, `surface_load_nodes.csv` and
`surface_load_summary.json`. It records `T_eq`, `T_cold`, cold/RE gas-heating
anchors, conductance/source, projection diagnostics and hashes for both
case/data pairs. The current Mechanical summary model version is
`6.0-aerodynamic-surface-load`.

Under `thermal_analysis/`:

| File | Physical content |
|---|---|
| `wall_temperature_history.csv` | Solved wall samples at original CFD boundary vertices versus `time_s`; temperatures in K, coordinates in m. No prescribed-temperature column. |
| `temperature_history.csv` | All-solid nodal minimum, arithmetic mean and maximum temperature in K versus seconds. |
| `surface_heat_flux_history.csv` | `convective_heat_flux_into_solid_W_m2`, `radiative_heat_flux_out_W_m2`, `net_heat_flux_into_solid_W_m2` versus seconds and surface index. |
| `surface_energy_history.csv` | `cumulative_convective_energy_J_m2`, `cumulative_radiative_energy_J_m2`, `cumulative_net_energy_J_m2` at each surface index. |
| `global_energy_history.csv` | Spatially integrated `convective_power_per_unit_span_W_m`, `radiative_power_per_unit_span_W_m`, `net_power_per_unit_span_W_m` and corresponding `cumulative_*_energy_per_unit_span_J_m`. |
| `thermal_nodes_final.csv` / `thermal_snapshots.npz` | Final nodal temperatures / selected physical-time spatial fields. |

The local net energy is

\[
Q''(s,t)=\int_0^t q''_{net}(s,\tau)\,d\tau
\]

and is evaluated by trapezoidal integration over saved physical substeps,
including the finite heat flux at the initial uniform temperature. The
initial cumulative energy is zero. Global powers use five-point Gaussian
quadrature on all selected quadratic FE boundary edges, followed by the
same temporal integration. They are an independent postprocessed boundary
integral. A separate `solid_energy_history.csv` integrates the actual quadratic
solid-element temperatures and compares stored-energy change with net surface
input at every native physical substep. `energy_balance.json` must pass before
the analysis is marked complete or eligible for comparison. Conservation uses
backward-Euler endpoint powers to match the explicit MAPDL `THETA=1` setting;
trapezoidal reporting integrals are retained separately. Native reaction heat
flows are not exported. See [the radiation correction](radiation_fix.md) for
the numerical tolerance, provenance changes, and thermal-only rerun commands.

Per-nose `reports/` and `runs/comparison/solid/` contain solved wall-temperature
curves, final wall/solid plots, local flux and energy histories and global
power/energy histories. Local comparison curves use arithmetic means of the
original wall samples, not exact boundary averages. Global histories retain
the solver's spatial integration. Only current packages and result hashes
are accepted by Mechanical reports.

To regenerate the plots from saved results without launching Ansys:

```powershell
.\.venv\Scripts\python.exe study.py report --runs-dir runs
```

Each nose's `reports/` and the compatible comparison group under
`runs/comparison/solid/` include:

- `surface_heat_flux_max_mean_history.png` and
  `surface_heat_load_max_mean_history.png`: maximum and simple wall-sample mean
  versus physical time, with separate convection, radiation and net panels.
- `final_surface_heat_flux_normalized_position.png` and
  `final_surface_heat_load_normalized_position.png`: final surface profiles on
  the same back → nose → back coordinate used for temperature comparisons.
  Per-nose plots also include profiles versus physical wall distance.
- `final_surface_heat_flux_maps.png` and `final_surface_heat_load_maps.png`:
  boundary sample maps, with common color scales across noses for each
  component. No interior flux or energy field is inferred from surface data.
- The existing global power/energy histories distinguish W/m and J/m of span
  from local W/m² and J/m². Heat load uses the saved time-integrated energy;
  it is not final heat flux multiplied by exposure time. Signed cooling is
  retained.

`max_temperature_comparison.png` and its CSV compare the maximum solid
temperature over **all saved physical times and all solid nodes**, including
the initial state. The CSV includes the time of the peak and the exposure
duration. This is not necessarily the final-time maximum. Wall-only historical
data are not substituted for a missing all-solid history. Incompatible flow,
material or duration settings continue to receive separate comparison groups.

`peak_heat_flux_comparison.png` compares each nose's maximum signed local
surface flux over every saved physical time and included wall sample, including
the initial state, in W/m². Convection, radiation and net heating have separate
panels; their peaks can occur at different times and locations, so peak
components are not expected to add/subtract to the peak net value. Its CSV
records the peak time and surface index for each component.

`total_heat_load_comparison.png` compares final accumulated boundary energy in
J/m of planar span using `global_energy_history.csv`. This is the integrated
energy over the analyzed boundary, not a local maximum/mean in J/m² or peak
flux multiplied by duration. Separate panels show incoming convective,
outgoing radiative and signed net energy. Both bar charts and their CSVs are
written to each nose's `reports/` and the compatible comparison folder when
the corresponding saved data exist. Negative net cooling remains negative.

`solver_temperature_history.csv` remains indexed by Fluent convergence
iteration. It is not physical time or an input to the transient solid solver.

## Insulated rear cut

The current template and updater select:

```yaml
calibration:
  exclude_rear_face: true
  rear_plane_tolerance_fraction: 1.0e-7
reporting:
  nose_direction: [1.0, 0.0]
```

The geometric nose direction is independent of the freestream direction. The
rear is the single connected chain of flat edges whose endpoints lie on the
minimum plane opposite that direction. The code fails if it cannot identify
that chain unambiguously; it never selects edges because their fitted
coefficients happen to be invalid. Split collinear rear edges are supported.

The full polygon remains closed in CFD and Mechanical. Rear faces are omitted
from the thermal fit, native Mechanical convection and radiation, surface
histories, surface-load plots and global heat-input integrals. The back has
natural zero-normal-flux boundary conditions: an **insulated cut**, with no
attachment/contact conduction. It can still warm through conduction within
the solid; its temperature is not constrained to 300 K. Solid temperature
fields and all-solid peak-temperature bars include every solid node.

At junctions with the top/bottom surfaces, nodal loading is projected only
from the included adjacent edges. Excluded rear data do not dilute those
values. Calibration diagnostics retain the original rear anchors and candidate
slopes, while reporting active and excluded failures separately. A bad fit on
any included edge still blocks thermal analysis.

Load CSVs retain full closed geometry and explicit `thermal_active` masks;
node CSVs also carry outgoing-edge `thermal_segment_active`. Zero load-law
placeholders at rear-only nodes are marked inactive and are not calibrated
coefficients. Surface-result CSVs omit rear-only nodes, retain the shared
corner nodes, and preserve outgoing-edge masks so plots never draw a heated
connection across the back.

CFD still contains its original rear wall and wake, and total drag/lift still
include all body-wall surfaces, including the rear. This thermal approximation
does not convert the model into an attached full vehicle. Set
`exclude_rear_face: false` explicitly to restore full-boundary thermal loading;
the updater preserves an explicit choice.

After updating, run the full batch with resume enabled. Valid primary CFD and
cold-wall calibration states/exports are reused when native inputs match;
thermal packages and Mechanical results are rebuilt for the new boundary
scope. Previous thermal results are archived. Changing the actual CFD inputs
still requires the corresponding native stages to run.

## Current approximations and validation limits

- One fixed flight condition and laminar planar CFD; no AoA/Mach sweep or
  CFD/solid feedback during the exposure. Drag/lift remain primary RE values.
- The gas-heating law is affine in wall temperature with fixed spatial
  coefficients. It extrapolates without clipping outside the calibration
  interval; a two-point fit does not establish accuracy there.
- Face-to-vertex projection and line interpolation introduce documented
  errors. Radiation of a projected temperature is not the projection of
  radiation at the original face temperatures; the imported equilibrium
  temperatures need not be exact stationary states of the projected law.
- Constant tungsten conductivity 170 W/(m·K), density 19,250 kg/m³ and heat
  capacity 130 J/(kg·K); constant emissivity 0.9. No oxidation, temperature
  dependent emissivity/material tables, ablation, melting, stress or creep.
- Filled planar solid reconstructed from CFD wall vertices, with unit span;
  no attached-airframe/contact thermal model or three-dimensional vehicle.
- Residual screens, mocked command tests and offline thermal/energy tests do
  not replace native engine verification, conservation checks or mesh,
  timestep, scheme and physical-property sensitivity studies.

The cloud environment has no licensed Fluent or Mechanical engines. Live
calibration and nonlinear thermal execution must be verified on Windows.

## Diagnose a rejected calibration without another CFD run

If package building reports that the cold/RE anchors do not identify positive
conductance, first inspect the existing matched exports:

```powershell
.\.venv\Scripts\python.exe study.py calibration-status --config ".\runs\sharp_short\_batch\config.yaml"
```

This command is read-only and does not launch Ansys. It verifies source hashes,
passed residual metadata, common geometry/mesh and physical face matching
before reporting candidate conductances. It prints the failed face count,
fraction of boundary length affected, and up to ten rejected faces with their
coordinates, cold and equilibrium temperatures, both heating anchors and the
signed candidate slope. Nonmatching or missing source files report
`status: unavailable`; incompatible anchors report `status: failed` and exit
with status 2.

Future package builds also save every matched face to
`thermal_bc/calibration_diagnostics.csv` and a summary to
`thermal_bc/calibration_diagnostics.json`, including when the load fit fails.
These diagnostics are not a usable Mechanical load package. No coefficient is
clipped, sign-flipped or replaced, and no failing faces are dropped.

For positive temperature separation, a nonpositive conductance means
`q_cold <= epsilon*sigma*(T_eq**4 - T_rad**4)` at that location. Two complete
steady wall-state CFD solutions need not provide a positive local slope:
changing the whole wall temperature can alter the boundary layer and wake.
Residual convergence alone does not validate this local surrogate. Diagnose
the affected locations and magnitudes before deciding whether field extraction,
numerical sensitivity or the heating approximation needs to change.
