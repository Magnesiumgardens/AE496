# Aerodynamic force outputs

The CFD force reports integrate every selected `body_wall` / `body_wall.*`
surface. They represent the complete planar body boundary, including its rear
closure. Drag is the pressure-plus-viscous force projected along the configured
freestream direction; lift uses its perpendicular direction. These are signed
sectional forces in N/m. They are not nose-tip-only, axisymmetric, or complete
3-D vehicle forces.

`aero_summary.json` retains `drag_per_unit_span_N_m` as total drag and also writes
`total_drag_per_unit_span_N_m`, `pressure_drag_per_unit_span_N_m`, and
`viscous_drag_per_unit_span_N_m`. The component values come from Fluent's native
force table and are checked against the independently reported total. Missing
or unparseable components remain null and are identified by `component_status`
and `component_error`; they are not replaced with zero.

No common reference width has been selected for this study. With
`aero.reference_area_per_unit_span_m: null`, `Cd`, `Cl`, `Cd_pressure`, and
`Cd_viscous` remain null. Compare sectional drag in N/m. When a common reference
width is explicitly supplied, the planar normalization is

\[
C_D = \frac{D'}{\tfrac12\rho_\infty V_\infty^2 b_{\rm ref}},
\]

where `D'` is N/m and `b_ref` is the reference area per unit span in metres.
Component coefficients use the same denominator. The reference length is
recorded separately; it does not enter this drag coefficient equation.

`solver_force_history.csv` records total drag and lift against the actual native
Fluent iteration counter. Its fields are `iteration`,
`drag_per_unit_span_N_m`, and `lift_per_unit_span_N_m`. It is a steady CFD
convergence diagnostic. It is unrelated to Mechanical's physical-time
`wall_temperature_history.csv`. The saved history is tied to the source
case/data and current conditions by hashes before reporting.

The default configuration samples the CSV every 25 iterations, retaining the
measured final sample even when Fluent stops between sampling intervals:

```yaml
aero:
  reference_area_per_unit_span_m: null
  force_history:
    enabled: true
    sample_every_iterations: 25
    recent_samples: 5
    min_samples: 5
    relative_tolerance: 0.01
    absolute_drag_tolerance_N_m: 1.0e-6
    require_stable: true
```

The recent five sampled drag values must pass both
`abs(D_final - mean(D_recent)) / abs(D_final) <= 0.01` and a peak-to-peak
variation of at most 1% of `abs(D_final)`. The second check prevents an
oscillating sequence from passing merely because its final value equals its
mean. Near zero drag, the configured absolute tolerance replaces division by
zero. Lift is recorded but is not an additional optimization objective or
acceptance threshold. These are sampled stability screens, not proof that
every possible unsteady fluctuation has been resolved.

The native raw report records every iteration. The final force state must
match the final native report calculation; actual iteration metadata is used
when available. Missing or inconsistent histories never receive a passing
status. Residual screening remains a separate requirement.

Updates retain existing CFD results. Their absent force histories cannot be
created from a final force value. To collect one from the developed solution:

```powershell
.\.venv\Scripts\python.exe study.py update
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe study.py continue --config ".\runs\blunt_1\_batch\config.yaml" --iterations 200 --stop-after aero
}
.\.venv\Scripts\python.exe study.py report --runs-dir runs
```

Continuation performs no initialization or numerical-model change. Two hundred
additional iterations are a recording budget, not a promise of convergence.
The last five samples still have to meet the configured tolerance, and the
residual screen must pass. This command stops before cold-wall calibration or
Mechanical. A later full study rebuilds downstream results if their source
CFD pair changed. New fresh solves record forces automatically.

When `require_stable: true`, failed, insufficient or missing histories stop
acceptance after `aero_summary.json` has been written. The measured forces,
temperature exports, and saved CFD pair remain available. Setting
`require_stable: false` allows diagnostic exports without claiming that their
forces are stable; reports retain their explicit status.

The comparison command creates `aerodynamic_comparison.csv`,
`force_iteration_history_comparison.csv`, optional force-history and drag
component figures, and per-case `reports/force_iteration_history.csv` and
`reports/force_iteration_history.png` when a valid history exists. These
tables preserve dimensional values for diagnostic review even if a stability
check is unavailable or fails. Every record carries `force_stability_status`.
Only passing drag-stability and residual checks permit Pareto screening eligibility;
plots mark other records separately. Passing that check alone does not prove
residual convergence, mesh independence, or the accuracy of the physical model.

The study does not add moments, angle-of-attack/Mach sweeps, or a weighted nose
selection score. The thermal calibration and Mechanical loading are unchanged
by the force reporting.
