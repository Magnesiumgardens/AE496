"""Shared Fluent startup and saved-case export; no duplicate launch scripts."""
from pathlib import Path

from .aero import aerodynamic_summary, reference_state, validate_saved_conditions
from .thermal_export import export_thermal_bc_package
from .utils import atomic_json, work_paths


def launch_solver(cfg, log_subdir="solver_logs"):
    import ansys.fluent.core as pyfluent
    runtime = cfg.get("runtime", {})
    logs = work_paths(cfg)["work"] / log_subdir
    logs.mkdir(parents=True, exist_ok=True)
    return pyfluent.launch_fluent(
        mode=pyfluent.FluentMode.SOLVER, dimension=pyfluent.Dimension.TWO,
        precision=pyfluent.Precision.DOUBLE,
        product_version=runtime.get("fluent_version", "25.1.0"),
        processor_count=int(runtime.get("processors", 4)),
        ui_mode=pyfluent.UIMode.GUI if runtime.get("fluent_gui", False) else pyfluent.UIMode.NO_GUI,
        cwd=str(logs), start_timeout=int(runtime.get("fluent_start_timeout_s", 300)),
    )


def export_saved_case(cfg, thermal=True, aero=True, inspect=False):
    paths = work_paths(cfg)
    for key in ("final_case", "final_data"):
        path = paths[key]
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"A nonempty solved case/data pair is required: {path}")
    if not cfg.get("solver", {}).get("run_iterations", True):
        raise ValueError("run_iterations=false cannot establish a solved CFD result")
    if aero:
        reference_state(cfg)
    solver = launch_solver(cfg, "export_logs")
    try:
        solver.settings.file.read_case_data(file_name=str(paths["final_case"]))
        if inspect:
            info = solver.fields.field_data.scalar_fields.allowed_values()
            fields = {name: {"field_name": name} for name in info
                      if "flux" in name.casefold() or "heat" in name.casefold()}
            atomic_json(paths["work"] / "available_heat_flux_fields.json", fields)
            print("[export] Heat/flux fields:", ", ".join(fields))
            return fields
        validate_saved_conditions(solver, cfg)
        base = cfg["geometry"].get("body_wall_name", "body_wall")
        walls = [z for z in solver.settings.setup.boundary_conditions.wall.get_object_names()
                 if z == base or z.startswith(base + ".")]
        result = {}
        if thermal:
            print("[export] Reading and saving body-wall heating...", flush=True)
            result["thermal_bc_package"] = export_thermal_bc_package(solver, cfg, walls)
            print("[export] Thermal BC saved:", result["thermal_bc_package"]["body_wall_faces_csv"], flush=True)
        if aero:
            try:
                print("[export] Computing body-wall forces from saved solution...", flush=True)
                result["aerodynamics"] = aerodynamic_summary(solver, cfg, walls)
                result["aerodynamics"]["source_case"] = str(paths["final_case"])
                atomic_json(paths["work"] / "aero_summary.json", result["aerodynamics"])
                print("[export] Drag:", result["aerodynamics"]["drag_per_unit_span_N_m"], "N/m", flush=True)
            except Exception as exc:
                atomic_json(paths["work"] / "aero_summary.json", {"status": "failed", "error": str(exc)})
                print("[export] Aerodynamic export failed; any completed thermal BC export is retained.", flush=True)
                raise
        return result
    finally:
        solver.exit()


def export_cli(thermal=True, aero=True):
    import argparse
    from .config import load_config
    parser = argparse.ArgumentParser(description="Export saved CFD wall heating and aerodynamic forces")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--inspect-fields", action="store_true")
    args = parser.parse_args()
    result = export_saved_case(load_config(args.config), thermal=thermal, aero=aero, inspect=args.inspect_fields)
    print("[export] Complete:", result)
