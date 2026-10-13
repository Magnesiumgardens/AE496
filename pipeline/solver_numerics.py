"""Explicit density-based solver numerics and freestream initialization."""
from .aero import reference_state


def configure_numerics(solver,cfg):
    options = cfg["solver"]
    solution = solver.settings.solution
    courant = float(options.get("courant_number", .1))
    solution.controls.courant_number = courant
    schemes = solution.methods.spatial_discretization.discretization_scheme
    if "flow" not in schemes.get_object_names():raise ValueError("Density-based flow discretization setting is unavailable")
    wanted = options.get("flow_scheme", "first-order-upwind")
    allowed = schemes["flow"].allowed_values()
    selected = next((value for value in allowed if str(value).casefold() == wanted.casefold()),None)
    if selected is None:raise ValueError(f"Requested flow scheme {wanted!r} unavailable: {allowed}")
    schemes["flow"] = selected
    print(f"[solve] Numerics: Courant={courant:g}; flow={selected}",flush=True)
    return dict(courant_number=courant,flow_scheme=selected)


def initialize_flow(solver,cfg,farfield):
    initialization = solver.settings.solution.initialization
    method = cfg["solver"].get("initialization", "freestream")
    if method == "hybrid":
        initialization.hybrid_initialize()
        return {"method":"hybrid"}
    if method != "freestream":raise ValueError("solver.initialization must be freestream or hybrid")
    initialization.compute_defaults(from_zone_type="pressure-far-field",from_zone_name=farfield)
    # Compute-from-farfield provides all native defaults. Explicitly set the
    # static thermodynamic state and velocity to avoid a low-Mach hybrid field.
    state = reference_state(cfg)
    defaults = initialization.defaults
    values = {"pressure":float(cfg["flight"]["static_pressure_pa"])-float(cfg["physics"].get("operating_pressure_pa",0)),
              "temperature":state["static_temperature_K"],
              "x-velocity":state["velocity_m_s"]*state["drag_direction"][0],
              "y-velocity":state["velocity_m_s"]*state["drag_direction"][1]}
    available = defaults.get_object_names()
    for key,value in values.items():
        if key not in available:raise ValueError(f"Fluent initialization default {key!r} is unavailable: {available}")
        defaults[key]=value
    initialization.standard_initialize()
    print(f"[solve] Initialized at freestream Mach {state['mach']:g}, T={state['static_temperature_K']:g} K",flush=True)
    return dict(method="freestream",**values)
