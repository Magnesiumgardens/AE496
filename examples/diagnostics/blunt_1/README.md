# Observed CFD startup run — not converged

These residuals were extracted from the supplied Fluent log for blunt_1: 1200 steady iterations, freestream initialization, first-order coupled flow, Courant number 0.1, adiabatic body wall. No values were reconstructed from requested iteration budgets.

The run completed without temperature/pressure limiting, but its final residuals exceed the screening thresholds. All four residuals have lower median values in iterations 901–1200 than in 601–900. This supports further numerical continuation; it does not validate temperature or force predictions, mesh independence, or material survivability.

The CSV, PNG and JSON under solver_diagnostics preserve the reported residuals and the screening result. These are diagnostic run data, not a converged nose-comparison result.
