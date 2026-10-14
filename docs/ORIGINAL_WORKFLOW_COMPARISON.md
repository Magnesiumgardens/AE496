# Original evidence and restored endpoint workflow

The supplied Python files and YAML establish the intended workflow, but they disagree in places: the uploaded thermal runner applies fixed heat flux, while the newer thermal YAML declares endpoint-derived convection. The separate adiabatic-reference runner and complete matching endpoint logs were not provided. The restored implementation follows the newer convection configuration and the user's current instruction to recover that strategy.

| Choice | Uploaded original | Restored default |
|---|---|---|
| First CFD endpoint | Prescribed 300 K wall; exports cold-wall heat flux | Same cold-wall reference, with explicit condition and numerical checks |
| Adiabatic reference | Configuration declares 600 iterations starting from the cold-wall field without initialization | Read the screened cold-wall case/data, set zero wall heat flux and continue without initialization |
| Mechanical loading | Newer thermal YAML declares `h(s) = q_cold(s)/(T_aw(s) − 300 K)` | Derive a spatial convection package from matched wall samples; load `h(s)` and recovery temperature |
| Flight direction | Earlier template uses +X | Confirmed −X into the current +X nose |
| CFD startup | Hybrid initialization; no explicit Courant/discretization override | Hybrid for cold wall, recorded native defaults unless overridden; no initialization for the warm adiabatic solve |
| Gas model | Laminar, ideal-gas density, Sutherland viscosity | Same configured baseline |
| Mesh | Nominal sizes and inflation; some batch paths could retry without inflation | Preserve an unchanged mesh; fail and diagnose rather than silently removing layers |
| Solid material | Constant tungsten `k=170`, `rho=19250`, `Cp=130` | Same configurable approximation |
| Exposure | 400 seconds; radiation enabled | Configurable 200-second default; radiation enabled |
| Aerodynamics | Original files do not include the current force comparison | Keep sectional drag/lift and optional common-reference coefficients from the adiabatic endpoint |
| Acceptance | Save/export after an iteration call; relied on native Fluent stopping | Align native and external targets, check limiting and provenance, block invalid Mechanical inputs |

The earlier uploaded `thermal_analysis_old.yaml` describes fixed-heat-flux solid loading. The newer `thermal_analysis.yaml` describes endpoint-derived convection. The restored workflow follows the latter: cold-wall flux calibrates `h`, while Mechanical computes the changing convection load as the solid warms.

## What differs from the intervening rebuild

The intervening direct-adiabatic rebuild removed the cold-wall endpoint and required a user-specified uniform coefficient. It also changed initialization, imposed Courant/first-order controls and tightened the energy residual target to `1e-6`. Those changes were not an equivalent implementation of the declared original strategy.

The restored default starts from the cold-wall solution, derives spatial `h`, and uses explicit `1e-3` baseline residual targets including energy. This intentionally restores the observed stopping scale; it does not establish better temperature accuracy. Actual recovery temperatures, forces and conservation must still settle, and numerical sensitivity remains part of CDR validation.

The independent direct-adiabatic/specified-h path is retained for compatibility and for cases with a justified externally supplied coefficient. Neither path invents temperature dependence of `h`. Endpoint-derived coefficients apply to the cold-wall reference and configured gas model; extending them to a hot solid or changing flight envelope requires a correlation or additional calibrated conditions.

## Documentation scope and runtime evidence

The fetched, tracked `main` documentation contained only a brief README. This audit covers Git source history and the uploaded files and handoffs; it does not establish the contents of GitHub issue comments or other untracked discussions. A historical plot or a case that reaches an iteration limit is not, by itself, proof of a converged adiabatic reference. Earlier logs showing repeated 5000 K temperature limiting cannot establish a trustworthy thermal boundary condition.

Fluent's displayed `energy` column is an equation residual, not a temperature in kelvin or a temperature error estimate. Optional iteration-temperature histories monitor the steady CFD calculation. Mechanical alone supplies the physical-time tungsten heating histories.

Windows failures involving settings adapters and optional raw-history `.out` locks were implementation problems in the rebuild. The restored workflow keeps candidate case/data recoverable and prevents a raw-log lock from undoing a valid solution. These file-handling fixes are retained independently of the physical-model restoration.
