# Original evidence and the selected single-solve model

The latest user-approved model replaces the active two-endpoint strategy with
one radiative-equilibrium CFD solve and explicitly prescribed surface-temperature
loading in Mechanical. That approval supersedes the earlier instruction to
derive `h` from cold-wall and adiabatic endpoints.

The uploaded original `pipeline/solve.py` contains a radiative-equilibrium
branch. It selects Fluent's wall `Radiation` thermal condition and supplies
external emissivity and surroundings temperature. The original configuration
describes this as solving wall temperature from gas-side heating and external
radiation. The current adapter uses the live allowed enum and explicit
option/value fields with readback checks for Ansys 2025 R1.

The approximate surface balance is
`q_gas = emissivity × sigma × (Tw⁴ − T_environment⁴)`, with no wall storage or
lateral solid conduction in this CFD wall model. The result `Tw(s)` is an
equilibrium surface temperature. It is not adiabatic recovery temperature or a
gas temperature sampled at a specified offset distance. Separate exported
fluid-side static samples retain their own field/sampling descriptions.

| Choice | Evidence from uploaded originals | Selected implementation |
|---|---|---|
| CFD thermal condition | Fixed-temperature and radiative-equilibrium branches are implemented | One `Radiation` wall solve; no prescribed 300 K temperature or zero-flux wall |
| Equilibrium inputs | Emissivity 0.90, radiation environment 3 K | Configurable same baseline; 300 K is reference metadata only |
| Mechanical coupling | Supplied runner actually applies fixed `HFLUX` plus `RAD`; no recovered temperature coupling | New, explicit nodal prescribed-temperature boundary, without extra heat-transfer loads |
| Cold/adiabatic calibration | Other uploaded YAML declared an endpoint-derived convection model | Retained compatibility implementation; bypassed by the active single-solve model |
| Direction and launch | Earlier template uses +X and automatic Geometry launch | User-confirmed −X into the +X nose; working SpaceClaim selection retained |
| CFD startup | Hybrid, steady density-based implicit, no explicit Courant/scheme override | Recorded native defaults unless user controls are explicit; migration preserves controls |
| Mesh and gas | Mach 8, p∞ 1090 Pa, T∞ 226.5 K; planar surface, laminar ideal/Sutherland baseline | Same configured baseline; retain repaired Ansys 2025 R1 meshing and unchanged mesh |
| Solid material | Constant tungsten `k=170`, `rho=19250`, `Cp=130` | Same configurable approximations, requiring high-temperature verification |
| Exposure | Historical runs use 400 seconds | Configurable 200-second current default; preserve user duration |
| Outputs | Temperature summaries, solid fields, histories and animation | Retain equivalent temperature visuals and sectional drag/lift, with explicit model labels |

## Meaning of prescribed Mechanical loading

Mechanical reconstructs the solid boundary from actual CFD wall vertices and
linearly maps their equilibrium temperatures to FE boundary nodes. The surface
temperature steps from its initial value to the specified spatial field; the
interior then changes by conduction. Wall history after application records
the imposed temperature, not a prediction of surface warm-up. Whole-solid
statistics include constrained boundary nodes as well as interior nodes.

The selected approximation does not derive `h`, calculate transient gas heating,
or provide exported constraint-reaction heat load. Convection or another
heat-transfer law would be needed to predict the surface's own warming; coupled
CFD/solid analysis would also allow feedback. Additional convection, radiation
or heat-flux loads on the prescribed boundary would change the selected model
and are excluded. CFD wall radiation remains active in the upstream equilibrium
temperature calculation.

## Evidence and validation scope

The fetched tracked `main` documentation had only a brief README. This audit
covers source history, uploaded Python/YAML, handoffs and supplied console logs;
it does not establish untracked GitHub discussions. The supplied successful
1161-iteration regression is a **300 K fixed-temperature run**, not a validated
radiative-equilibrium run. The original radiation branch documents and
implements the capability; its live execution still needs verification.

Historical 5000 K clipping, an iteration budget completing, or a saved plot
does not establish a trustworthy thermal field. Current export and Mechanical
loading check the intended wall model, residual screen and source hashes.
Older endpoint results remain retained but cannot substitute for the selected
single radiating-wall solution.

Fluent's `energy` column is an equation residual, not temperature in kelvin.
Optional CFD temperature histories use iteration labels and diagnose a steady
calculation. Mechanical alone supplies physical-time solid histories in this
workflow. Its prescribed surface curves and computed interior conduction are
labelled separately.

Adapter, history-file locking, candidate recovery, configuration migration,
mesh reuse and aerodynamic-report fixes are retained independently of this
model change. Offline tests validate code contracts and plots; they do not
validate licensed Ansys numerical results, mesh independence or material
survivability for CDR.
