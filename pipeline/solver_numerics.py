"""Explicit density-based solver numerics and freestream initialization."""
import math

from .aero import reference_state


def configure_numerics(solver,cfg):
    options = cfg["solver"]
    solution = solver.settings.solution
    courant = float(options.get("courant_number", .1))
    solution.controls.courant_number = courant
    schemes = solution.methods.spatial_discretization.discretization_scheme
    wanted = options.get("flow_scheme", "first-order-upwind")
    # Fluent 251's creatable dictionary may initially have no entries. Native
    # assignment creates the entry; get_object_names is not a capability list.
    try:
        schemes["flow"] = wanted
        entry = schemes["flow"]
        selected = entry.get_state()
        if selected != wanted:
            raise ValueError(f"Fluent returned {selected!r} after requesting {wanted!r}")
    except (RuntimeError, ValueError, KeyError, AttributeError) as exc:
        try:allowed = schemes["flow"].allowed_values()
        except (RuntimeError, KeyError, AttributeError):allowed = []
        raise ValueError(f"Could not set density-based flow scheme {wanted!r}; "
                         f"native entries={schemes.get_object_names()}, allowed flow choices={allowed}: {exc}") from exc
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
    for key,value in values.items():
        # Initialization defaults are also a creatable dictionary in Fluent 251.
        defaults[key] = value
        actual = float(defaults[key].get_state())
        if not math.isclose(actual,value,rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError(f"Fluent initialization default {key!r}: requested {value:g}, returned {actual:g}")
    initialization.standard_initialize()
    print(f"[solve] Initialized at freestream Mach {state['mach']:g}, T={state['static_temperature_K']:g} K",flush=True)
    return dict(method="freestream",**values)
