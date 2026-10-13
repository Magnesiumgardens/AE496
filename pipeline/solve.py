from __future__ import annotations

from pathlib import Path
import json

from .aero import reference_state, report_scalar
from .cfd import launch_solver

from .utils import work_paths
from .utils import file_sha256
from .temperature_export import atomic_csv


class SolverPipelineError(RuntimeError):
    pass


def _ensure_zone_type(bc, name: str, new_type: str) -> None:
    attr = new_type.replace("-", "_")
    collection = getattr(bc, attr, None)
    if collection is not None:
        try:
            if name in collection.get_object_names():
                return
        except Exception:
            pass
    bc.set_zone_type(zone_list=[name], new_type=new_type)


def _zone_names(bc) -> dict[str, list[str]]:
    out = {}
    for key in ("wall", "pressure_far_field", "pressure_outlet", "interior"):
        obj = getattr(bc, key, None)
        if obj is None:
            continue
        try:
            out[key] = list(obj.get_object_names())
        except Exception:
            pass
    return out
def _matching_zone_names(bc, base_name: str) -> list[str]:
    """Return all Fluent zones belonging to a named-selection base name."""
    matches = []

    for names in _zone_names(bc).values():
        for name in names:
            if name == base_name or name.startswith(base_name + "."):
                matches.append(name)

    return sorted(set(matches))

def _set_wall_thermal(wall, thermal_cfg: dict) -> dict:
    if thermal_cfg != {"mode": "adiabatic"}:
        raise SolverPipelineError("This study requires an adiabatic wall with no thermal loading")
    wall.thermal.thermal_condition = "Heat Flux"
    wall.thermal.heat_flux.option = "value"
    wall.thermal.heat_flux.value = 0.0
    return {"thermal_condition": "Heat Flux", "heat_flux_W_m2": 0.0}


def _temperature_summary(solver, wall_names: list[str]) -> dict:
    reports = solver.settings.solution.report_definitions.surface

    specs = {
        "wall_temperature_min_k": ("auto-wall-temp-min", "surface-facetmin"),
        "wall_temperature_avg_k": ("auto-wall-temp-avg", "surface-areaavg"),
        "wall_temperature_max_k": ("auto-wall-temp-max", "surface-facetmax"),
    }

    for _, (name, rtype) in specs.items():
        reports[name] = {
            "surface_names": wall_names,
            "report_type": rtype,
            "field": "temperature",
        }

    compute = solver.settings.solution.report_definitions.compute
    return {key: report_scalar(compute(report_defs=[name]), name)
            for key, (name, _) in specs.items()}


def _iterate_with_temperature_history(solver, cfg, wall_names):
    """Sample steady temperature convergence only with a measured iteration counter."""
    import math
    from numbers import Real
    iterations = int(cfg["solver"].get("max_iterations", 1200))
    options = cfg["solver"].get("temperature_history", {})
    iterate = solver.settings.solution.run_calculation.iterate
    if not options.get("enabled", True):
        iterate(iter_count=iterations)
        return {"status": "disabled"}
    interval = int(options.get("sample_every_iterations", 25))
    rows, requested = [], 0
    try:
        available = solver.rp_vars.allowed_values()
        counter = next((name for name in ("iteration", "iterations", "iteration-count", "global-iteration")
                        if name in available), None)
        if counter is None:raise ValueError("Fluent exposes no supported actual iteration counter")
        def sample():
            value = solver.rp_vars(counter)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0 or not float(value).is_integer():
                raise ValueError("Invalid actual iteration counter")
            temperatures = _temperature_summary(solver, wall_names)
            if not (0 < temperatures["wall_temperature_min_k"] <= temperatures["wall_temperature_avg_k"]
                    <= temperatures["wall_temperature_max_k"]):
                raise ValueError("Nonphysical or inconsistent wall-temperature sample")
            return dict(iteration=int(value), **temperatures)
        rows.append(sample())
    except Exception as exc:
        print(f"[temperature-history] Unavailable: {exc}; continuing the steady solve.", flush=True)
        iterate(iter_count=iterations)
        return {"status": "unavailable", "error": str(exc)}
    status, error = "complete", None
    while requested < iterations:
        chunk = min(interval, iterations - requested)
        iterate(iter_count=chunk)
        requested += chunk
        try:
            row = sample()
            advanced = row["iteration"] - rows[-1]["iteration"]
            if advanced <= 0 or advanced > chunk:raise ValueError("Actual iteration counter did not advance as requested")
            rows.append(row)
            if advanced < chunk:break  # Fluent stopped this chunk early.
        except Exception as exc:
            status, error = "partial", str(exc)
            print(f"[temperature-history] Capture stopped: {exc}; retaining measured samples.", flush=True)
            if requested < iterations:iterate(iter_count=iterations - requested)
            break
    path = work_paths(cfg)["work"] / "solver_temperature_history.csv"
    try:
        atomic_csv(path, rows)
        digest = file_sha256(path)
    except OSError as exc:
        print(f"[temperature-history] Could not save optional history: {exc}; preserving the CFD solution.", flush=True)
        return {"status": "unavailable", "error": str(exc)}
    return dict(status=status, error=error, csv=path.name, csv_sha256=digest,
                sample_count=len(rows), iteration_counter=counter,
                interpretation="Steady CFD convergence versus measured Fluent iteration; not physical time.")


def solve_case(cfg: dict, mesh_file: str | Path) -> dict:
    paths = work_paths(cfg)

    # Use absolute paths because Fluent will run from ./logs.
    mesh_file = Path(mesh_file).resolve()
    if not mesh_file.is_file():
        raise SolverPipelineError(f"Mesh does not exist: {mesh_file}")
    if cfg.get("aero", {}).get("enabled", True):
        reference_state(cfg)  # Validate reference assumptions before launching Fluent.

    print("[solve] Launching 2-D double-precision Fluent solver...")
    solver = launch_solver(cfg)

    summary = {}

    try:
        print(f"[solve] Reading 2-D mesh: {mesh_file}")
        solver.settings.file.read_mesh(file_name=str(mesh_file))

        print("[solve] Checking mesh...")
        solver.tui.mesh.check()

        setup = solver.settings.setup
        setup.general.solver.type = cfg["solver"].get(
            "density_solver", "density-based-implicit"
        )
        setup.general.solver.time = "steady"

        # Adiabatic walls do not disable the fluid energy equation. Compression
        # and viscous effects still determine the hot gas/recovery temperatures.
        energy = setup.models.energy
        energy.enabled = True
        setup.models.radiation.model = "none"
        try:
            if energy.viscous_dissipation.is_active():
                energy.viscous_dissipation = True
        except Exception:
            pass

        setup.models.viscous.model = cfg["physics"].get("viscous_model", "laminar")

        air_cfg = cfg["physics"]["air"]
        air = setup.materials.fluid["air"]
        air.density.option = air_cfg.get("density_model", "ideal-gas")
        air.viscosity.option = air_cfg.get("viscosity_model", "sutherland")
        if air_cfg.get("viscosity_model", "sutherland") == "sutherland":
            air.viscosity.sutherland.option = "three-coefficient-method"
            air.viscosity.sutherland.reference_viscosity = float(
                air_cfg["sutherland_reference_viscosity"]
            )
            air.viscosity.sutherland.reference_temperature = float(
                air_cfg["sutherland_reference_temperature_k"]
            )
            air.viscosity.sutherland.effective_temperature = float(
                air_cfg["sutherland_effective_temperature_k"]
            )

        setup.general.operating_conditions.operating_pressure = float(
            cfg["physics"].get("operating_pressure_pa", 0.0)
        )

        wall_name = cfg["geometry"].get("body_wall_name", "body_wall")
        far_name = cfg["geometry"].get("farfield_name", "farfield")
        bc = setup.boundary_conditions

        print(f"[solve] Boundary zones before correction: {_zone_names(bc)}")

        # PyFluent/Fluent may split one named selection into multiple edge zones,
        # for example body_wall.1 ... body_wall.4.
        wall_zones = _matching_zone_names(bc, wall_name)
        far_zones = _matching_zone_names(bc, far_name)

        if not wall_zones:
            raise SolverPipelineError(
                f"No Fluent boundary zones matched '{wall_name}'. "
                f"Available zones: {_zone_names(bc)}"
            )

        if not far_zones:
            raise SolverPipelineError(
                f"No Fluent boundary zones matched '{far_name}'. "
                f"Available zones: {_zone_names(bc)}"
            )

        print(f"[solve] Body-wall zones: {wall_zones}")
        print(f"[solve] Farfield zones: {far_zones}")

        # Make sure every matching zone has the intended boundary type.
        for zone_name in wall_zones:
            _ensure_zone_type(bc, zone_name, "wall")

        for zone_name in far_zones:
            _ensure_zone_type(bc, zone_name, "pressure-far-field")

        # Apply identical freestream conditions to every farfield edge.
        d = cfg["flight"]["flow_direction"]

        for zone_name in far_zones:
            far = bc.pressure_far_field[zone_name]

            # Fluent requires gauge pressure: p_static minus operating pressure.
            far.momentum.gauge_pressure = float(cfg["flight"]["static_pressure_pa"]) - float(
                cfg["physics"].get("operating_pressure_pa", 0.0)
            )
            far.momentum.mach_number = float(
                cfg["flight"]["mach"]
            )
            far.thermal.temperature = float(
                cfg["flight"]["static_temperature_k"]
            )
            far.momentum.flow_direction[0] = float(d[0])
            far.momentum.flow_direction[1] = float(d[1])

        # Apply the thermal wall condition to every vehicle-wall edge.
        wall_thermal_setup = {}

        for zone_name in wall_zones:
            wall = bc.wall[zone_name]
            wall_thermal_setup[zone_name] = _set_wall_thermal(
                wall,
                cfg["thermal"],
            )

        summary["thermal_mode"] = "adiabatic"
        summary["wall_thermal_setup"] = wall_thermal_setup
        summary["body_wall_zones"] = wall_zones
        summary["farfield_zones"] = far_zones

        print(f"[solve] Boundary zones after correction: {_zone_names(bc)}")

        if bool(cfg.get("solver", {}).get("hybrid_initialize", True)):
            print("[solve] Hybrid initialization...")
            solver.settings.solution.initialization.hybrid_initialize()

        print(f"[solve] Writing setup case: {paths['setup_case']}")
        solver.settings.file.write(file_name=str(paths["setup_case"]), file_type="case")

        ran_iterations = bool(
            cfg["solver"].get("run_iterations", True)
        )

        if ran_iterations:
            n = int(cfg["solver"].get("max_iterations", 1200))

            print(f"[solve] Running up to {n} steady iterations...")

            summary["temperature_history"] = _iterate_with_temperature_history(solver, cfg, wall_zones)

        else:
            print(
                "[solve] run_iterations=false: "
                "setup completed, solve skipped."
            )

        summary["solution_status"] = "iterations_completed" if ran_iterations else "setup_only"
        summary["convergence_status"] = "not_verified"
        # Post-processing is a separate saved-case stage. Persist the solution
        # before any field/report adapter can fail.

        if bool(cfg.get("outputs", {}).get("write_case_data", True)):
            print(f"[solve] Writing final case/data: {paths['final_case']}")
            solver.settings.file.write_case_data(file_name=str(paths["final_case"]))
            if summary.get("temperature_history", {}).get("status") in ("complete", "partial"):
                summary["temperature_history"]["source_sha256"] = {
                    key: file_sha256(paths[key]) for key in ("final_case", "final_data")}

        paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        solver.exit()
