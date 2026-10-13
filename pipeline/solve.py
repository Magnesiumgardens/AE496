from __future__ import annotations

from pathlib import Path
import json

from .aero import aerodynamic_summary, reference_state, report_scalar
from .cfd import launch_solver

from .thermal_export import export_thermal_bc_package
from .utils import work_paths


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
    # This is the CFD reference boundary, not the transient solid temperature.
    if thermal_cfg["mode"] != "fixed_temperature":
        raise SolverPipelineError("Use a fixed-temperature CFD heating reference")
    temperature = float(thermal_cfg["wall_temperature_k"])
    wall.thermal.thermal_condition = "Temperature"
    wall.thermal.temperature = temperature
    return {"thermal_condition": "Temperature", "wall_temperature_k": temperature}


def _temperature_summary(solver, wall_names: list[str]) -> dict:
    reports = solver.settings.solution.report_definitions.surface

    specs = {
        "wall_temperature_min_k": ("auto-wall-temp-min", "surface-facetmin"),
        "wall_temperature_avg_k": ("auto-wall-temp-avg", "surface-areaavg"),
        "wall_temperature_max_k": ("auto-wall-temp-max", "surface-facetmax"),
    }

    report_names = []

    for _, (name, rtype) in specs.items():
        try:
            if name in reports.get_object_names():
                reports.delete(names=[name])
        except Exception:
            pass

        reports[name] = {
            "surface_names": wall_names,
            "report_type": rtype,
            "field": "temperature",
        }

        report_names.append(name)

    raw = solver.settings.solution.report_definitions.compute(
        report_defs=report_names
    )

    return {key: report_scalar(raw, name) for key, (name, _) in specs.items()}


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

        energy = setup.models.energy
        energy.enabled = True
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

        summary["cfd_reference_wall_temperature_K"] = cfg["thermal"].get("wall_temperature_k")
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

            solver.settings.solution.run_calculation.iterate(
                iter_count=n
            )

        else:
            print(
                "[solve] run_iterations=false: "
                "setup completed, solve skipped."
            )

        # Only generate solution-derived reports after an actual solve.
        summary["solution_status"] = "iterations_completed" if ran_iterations else "setup_only"
        summary["convergence_status"] = "not_verified"
        if ran_iterations:
            if cfg.get("aero", {}).get("enabled", True):
                try:
                    summary["aerodynamics"] = aerodynamic_summary(solver, cfg, wall_zones)
                except Exception as exc:
                    summary["aerodynamics"] = {"status": "failed", "error": str(exc)}
                    print(f"[aero] ERROR: {exc}")

            if bool(
                cfg.get("outputs", {}).get(
                    "compute_temperature_summary",
                    True,
                )
            ):
                try:
                    summary.update(
                        _temperature_summary(
                            solver,
                            wall_zones,
                        )
                    )
                except Exception as e:
                    summary["temperature_summary_error"] = str(e)

            # Export teammate-ready aerothermal boundary conditions.
            if bool(
                cfg.get("thermal_export", {}).get(
                    "enabled",
                    True,
                )
            ):
                try:
                    print(
                        "[thermal-export] Exporting aerothermal "
                        "boundary-condition package..."
                    )

                    summary["thermal_bc_package"] = (
                        export_thermal_bc_package(
                            solver=solver,
                            cfg=cfg,
                            wall_zones=wall_zones,
                        )
                    )

                except Exception as e:
                    summary["thermal_bc_export_error"] = str(e)

                    print(
                        "[thermal-export] ERROR: "
                        f"{e}"
                    )

            # Optional legacy Fluent-native heat-flux export.
            if bool(
                cfg.get("outputs", {}).get(
                    "export_custom_heat_flux",
                    False,
                )
            ):
                try:
                    solver.settings.file.export.custom_heat_flux(
                        file_name=str(paths["heat_flux"]),
                        wall_function=False,
                        surface_name_list=wall_zones,
                    )

                    summary["custom_heat_flux_file"] = str(
                        paths["heat_flux"]
                    )

                except Exception as e:
                    summary["custom_heat_flux_export_error"] = str(e)

        else:
            summary["solution_postprocessing"] = (
                "Skipped because run_iterations=false."
            )

        if bool(cfg.get("outputs", {}).get("write_case_data", True)):
            print(f"[solve] Writing final case/data: {paths['final_case']}")
            solver.settings.file.write_case_data(file_name=str(paths["final_case"]))

        if "aerodynamics" in summary:
            # Save after final case/data so resume can detect stale force outputs.
            (Path(cfg["project"]["work_dir"]) / "aero_summary.json").write_text(
                json.dumps(summary["aerodynamics"], indent=2), encoding="utf-8"
            )
        paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        solver.exit()
