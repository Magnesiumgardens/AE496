# Uploaded original and current workflows

This comparison describes the supplied code and configuration, not proof that every declared stage ran. The original adiabatic-reference runner, thermal runner and matching per-case solver logs were not included.

| Choice | Uploaded original | Current version |
|---|---|---|
| Primary CFD | Prescribed 300 K wall, exporting cold-wall heat flux | Direct adiabatic wall, exporting gas/recovery temperatures |
| Adiabatic reference | Config declares a separate 600-iteration reference from the converged cold-wall field, without initialization | Freestream startup, then continuation of the saved adiabatic field |
| Convection coefficient | Thermal YAML declares h(s)=q_cold(s)/(T_aw(s)-300 K) from two CFD endpoints | Required specified uniform h; no cold-wall endpoint |
| Flow direction | Template +X | Confirmed -X for current noses |
| Numerics | Hybrid initialization; Courant/discretization not explicitly set | Freestream initialization, explicit Courant and first-order coupled flow |
| Gas model | Laminar, ideal-gas density, Sutherland viscosity | Same configured choices |
| Mesh | Same nominal sizes/inflation parameters; batch could retry without inflation | Nominal settings match, but actual meshes cannot be assumed identical |
| Solid | Constant tungsten k=170, rho=19250, Cp=130; 400 s; radiation enabled | Same configurable material constants; 200 s default; radiation off by default |
| Convergence | Primary solver saves/exports after its iterate call, without an independent residual screen | Residual screen plus provenance checks |

The older thermal YAML describes fixed-heat-flux loading. The newer uploaded thermal YAML describes endpoint-derived convection. Those are different solid boundary conditions.

The original main solver/exporter cannot by itself produce an adiabatic recovery temperature: its prescribed wall temperature is 300 K. The declared separate reference runner is essential, but its implementation was not uploaded. The previous successful geometry, flow direction, mesh and loading should be matched before treating old and current results as equivalent. The current user requirement for adiabatic CFD remains in force; the comparison does not restore cold-wall CFD.

## Latest runtime issue

Fluent announced convergence at iteration 3200 with energy residual 4.4518e-4; the script's energy screen was 1e-6. The updated code explicitly applies the same configurable absolute criteria in Fluent and in the screen, without resetting normalization.

The later Windows error concerned the optional raw iteration-history `.out` file. Promotion now copies candidate files instead of consuming them, and a raw-log lock cannot undo the solution. Candidate files remain recoverable if a critical promotion fails. The raw history is not a Mechanical boundary condition.
