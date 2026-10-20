# Radiation in the transient solid analysis

The calibrated aerodynamic surface-load model uses steady radiative-equilibrium
CFD and a cold-wall calibration to obtain a fixed spatial convection law. Mechanical
then solves the solid surface and interior temperatures over physical time:

\[
q''_{\mathrm{conv}}(s,T_w)=H(s)[T_{\mathrm{drive}}(s)-T_w],\qquad
q''_{\mathrm{rad}}(T_w)=\epsilon\sigma(T_w^4-T_{\mathrm{surroundings}}^4).
\]

The previous implementation assigned `SFL(...,"RAD",...)` to the solid boundary
without verifying whether radiation reached the solid elements. The uploaded
results showed a positive solid temperature rise alongside negative accumulated
net input. Recalculating a correct radiation formula in a CSV does not establish
that the same cooling participated in the native solve.

The updated model uses `PLANE293` solid elements and explicit radiation-capable
`SURF151` elements on the exposed boundary. Convection and radiation are evaluated
with the current solved surface temperature. The ambient reference node has its
configured surroundings temperature prescribed; **no solid surface node receives
a prescribed `TEMP` constraint** in this loading mode. An excluded rear cut remains
insulated. The ambient node is excluded from solid temperature statistics, solid
stored energy, and the physical solid mesh outputs.

The solid starts uniformly at the configured initial temperature (300 K in the
template). `transient.duration_s` remains configurable for 200 or 400 seconds.
The CFD and calibration fields remain fixed during this transient. This is not
two-way coupling to Fluent and does not add an AoA or Mach sweep.

## Energy verification and saved results

The updated model version is `7.0-surf151-aerodynamic-surface-load`. A result is
eligible for current solid comparisons only when its radiation configuration,
stored-energy verification, source provenance, and output hashes are current.
Existing version 6 thermal results cannot pass as corrected results.

`solid_energy_history.csv` records energy stored in the full solid relative to
its initial state, integrated over the quadratic solid finite elements. This is
separate from merely integrating the boundary load. `energy_balance.json` records
the comparison between that stored-energy change and net surface input. For the
configured backward-Euler transient integration, the check uses physical substep
endpoint powers. Reporting still uses trapezoidal time integration for the
existing heat-load curves; that distinction is recorded rather than hidden.
Every step and cumulative balance must agree within 2% of gross exchanged
energy (absolute convection plus absolute radiation), plus 0.001 J/m. This is
a numerical consistency screen, not a 2% claim of physical model accuracy.
Failed checks retain temperatures and energy diagnostics, do not mark the
analysis complete, and prevent its inclusion in current solid comparisons.
The check integrates FE temperatures and surface laws independently; it does
not substitute a native reaction-heat-flow measurement or mesh/time-step study.

The original result files remain available: `temperature_history.csv` contains
solid minimum/mean/maximum temperatures versus physical seconds;
`wall_temperature_history.csv` contains solved surface temperatures; and
`surface_heat_flux_history.csv`, `surface_energy_history.csv`, and
`global_energy_history.csv` contain local flux, local accumulated energy, and
integrated powers/energies per unit planar span. These are separate from Fluent
iteration histories, which describe steady-solution convergence.

## Rerun only the solid analysis

From the AE496 repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe study.py update
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe run_thermal_analysis.py --config ".\runs\sharp_long\_batch\config.yaml" --thermal-config ".\thermal_analysis.yaml"
}
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe study.py report --runs-dir ".\runs"
}
```

The thermal-only entry point reuses an existing valid calibration. It does not
remesh, solve Fluent, or recompute aerodynamic forces. Each other nose can be
rerun by substituting its case name. Preserve the configured emissivity and
surroundings temperature so the calibration anchors remain consistent.

## Remaining physical approximations

- Tungsten conductivity, density, and specific heat are constant configured
  values; the template uses 170 W/(m K), 19,250 kg/m³, and 130 J/(kg K).
- Emissivity is the configured constant shared with the CFD calibration. It is
  not automatically changed to 0.45 or another assumed material value.
- Radiation is to the specified uniform surroundings. Oxidation, evolving surface
  emissivity, participating gas radiation, and material phase changes are absent.
- The affine convection law is calibrated at two wall states and is not a new
  CFD solve at every solid temperature. Its effective driving temperature is a
  fitted parameter, not a directly sampled gas-temperature field.
- The model is a 2-D planar section with unit span, constant flight conditions,
  and any selected rear insulation. It does not represent full vehicle heat
  capacity, vehicle attachments, or a changing flight trajectory.

The cloud environment has no licensed native MAPDL engine. Offline checks cover
load construction, finite-element energy integration, failure detection, and
report eligibility. The corrected native solve must run on the licensed Windows
installation and satisfy its saved energy verification before use in comparison.
