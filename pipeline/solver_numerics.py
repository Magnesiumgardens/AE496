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
    names = schemes.get_object_names()
    # Fluent251 exposes the coupled density-based equations as amg-c. Other
    # releases may use flow. Generated CreatableNamedObjectMixin does not imply
    # that the server permits creating additional discretization entries.
    key = next((name for name in ("amg-c", "flow") if name in names), None)
    if key is None:
        raise ValueError(f"No existing coupled-flow discretization entry; native entries={names}")
    entry = schemes[key]
    allowed = entry.allowed_values()
    def canonical(value):return "".join(c for c in str(value).casefold() if c.isalnum())
    selected = next((value for value in allowed if canonical(value) == canonical(wanted)), None)
    if selected is None:
        raise ValueError(f"Flow scheme {wanted!r} unavailable on {key!r}; native choices={allowed}")
    schemes[key] = selected
    actual = schemes[key].get_state()
    if canonical(actual) != canonical(selected):
        raise ValueError(f"Fluent {key!r}: requested {selected!r}, returned {actual!r}")
    print(f"[solve] Numerics: Courant={courant:g}; flow[{key}]={selected}",flush=True)
    return dict(courant_number=courant,flow_scheme=selected,flow_scheme_entry=key)


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
    missing = set(values) - set(available)
    if missing:
        raise ValueError(f"Freestream initialization defaults missing {sorted(missing)}; native defaults={available}")
    for key,value in values.items():
        # Only update entries which the Fluent server actually exposes.
        defaults[key] = value
        actual = float(defaults[key].get_state())
        if not math.isclose(actual,value,rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError(f"Fluent initialization default {key!r}: requested {value:g}, returned {actual:g}")
    initialization.standard_initialize()
    print(f"[solve] Initialized at freestream Mach {state['mach']:g}, T={state['static_temperature_K']:g} K",flush=True)
    return dict(method="freestream",**values)
