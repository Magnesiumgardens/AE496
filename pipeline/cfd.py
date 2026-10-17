"""Shared Fluent startup and saved-case export; no duplicate launch scripts."""
from pathlib import Path

from .aero import aerodynamic_summary, reference_state, validate_saved_conditions
from .temperature_export import export_temperatures, temperature_paths
from .utils import atomic_json, file_sha256, work_paths


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
    endpoint = cfg.get("workflow",{}).get("strategy") == "two_endpoint"
    if endpoint and not cfg.get("_endpoint_kind"):
        from .endpoint_workflow import effective_cfg
        if inspect:return export_saved_case(effective_cfg(cfg,"adiabatic"),inspect=True)
        result = {}
        if thermal:
            result["cold_wall"] = export_saved_case(effective_cfg(cfg,"cold_wall"),thermal=False,aero=False)
        result.update(export_saved_case(effective_cfg(cfg,"adiabatic"),thermal=thermal,aero=aero))
        if thermal:
            from .endpoint_loading import build_convection_package
            result["convection"] = build_convection_package(cfg)
        return result
    paths = work_paths(cfg)
    for key in ("final_case", "final_data"):
        path = paths[key]
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"A nonempty solved case/data pair is required: {path}")
    if not cfg.get("solver", {}).get("run_iterations", True):
        raise ValueError("run_iterations=false cannot establish a solved CFD result")
    if aero:
        reference_state(cfg)
    if not inspect:
        from .solution_quality import require_usable_solution
        quality = require_usable_solution(cfg)
        if (endpoint or cfg.get("_calibration_kind")) and quality.get("status") != "residual_screen_passed":
            raise ValueError("Endpoint exports require a matching passed residual screen")
        if (cfg["thermal"].get("mode") == "radiative_equilibrium"
                and quality.get("status") != "residual_screen_passed"):
            raise ValueError("Radiative-equilibrium export requires a matching passed residual screen")
    solver = launch_solver(cfg, "export_logs")
    try:
        solver.settings.file.read_case_data(file_name=str(paths["final_case"]))
        if inspect:
            info = solver.fields.field_data.scalar_fields.allowed_values()
            fields = {name: {"field_name": name} for name in info
                      if any(word in name.casefold() for word in ("temp", "flux", "heat"))}
            atomic_json(paths["work"] / "available_temperature_fields.json", fields)
            print("[export] Temperature/flux fields:", ", ".join(fields))
            return fields
        validate_saved_conditions(solver, cfg)
        base = cfg["geometry"].get("body_wall_name", "body_wall")
        walls = [z for z in solver.settings.setup.boundary_conditions.wall.get_object_names()
                 if z == base or z.startswith(base + ".")]
        result = {}
        if endpoint and (thermal or cfg.get("_endpoint_kind") == "cold_wall"):
            from .endpoint_export import export_endpoint
            result["endpoint"] = export_endpoint(solver,cfg,walls)
        if cfg.get("_calibration_kind") == "cold_wall":
            from .endpoint_export import export_endpoint
            result["calibration"] = export_endpoint(solver, cfg, walls)
        elif cfg.get("calibration", {}).get("enabled", False) and cfg["thermal"]["mode"] == "radiative_equilibrium":
            from .aerothermal_calibration import export_re_calibration
            result["calibration"] = export_re_calibration(solver, cfg, walls)
        if thermal:
            mode = cfg["thermal"]["mode"]
            label = ("radiative-equilibrium wall and near-wall fluid temperatures"
                     if mode == "radiative_equilibrium" else "adiabatic gas/recovery temperatures")
            print(f"[export] Saving {label}...", flush=True)
            result["temperatures"] = export_temperatures(solver, cfg, walls)
            result["temperatures"]["source_sha256"] = {
                key: file_sha256(paths[key]) for key in ("final_case", "final_data")}
            result["temperatures"]["wall_csv_sha256"] = file_sha256(temperature_paths(cfg)[0])
            atomic_json(temperature_paths(cfg)[1], result["temperatures"])
            if mode == "radiative_equilibrium":
                # The exported field is authoritative when the optional native
                # report adapter could not provide the final solver summary.
                import json
                metadata = json.loads(paths["summary"].read_text(encoding="utf-8")) if paths["summary"].exists() else {}
                stats = result["temperatures"]["wall_temperature"]
                metadata.update(wall_temperature_min_k=stats["minimum_K"],
                                wall_temperature_avg_k=stats["length_weighted_mean_K"],
                                wall_temperature_max_k=stats["maximum_K"])
                metadata["wall_temperature_report"] = {"status": "complete", "source": "saved_boundary_temperature_field",
                    "meaning": "Steady radiative-equilibrium wall temperature; not adiabatic recovery temperature"}
                atomic_json(paths["summary"], metadata)
            print("[export] Temperatures saved:", result["temperatures"]["wall_temperatures_csv"], flush=True)
        if aero:
            try:
                print("[export] Computing body-wall forces from saved solution...", flush=True)
                result["aerodynamics"] = aerodynamic_summary(solver, cfg, walls)
                result["aerodynamics"]["source_case"] = str(paths["final_case"])
                result["aerodynamics"]["source_sha256"] = {key:file_sha256(paths[key]) for key in ("final_case","final_data")}
                destination = Path(cfg.get("_study_work_dir",paths["work"]))
                atomic_json(destination / "aero_summary.json", result["aerodynamics"])
                print("[export] Drag:", result["aerodynamics"]["drag_per_unit_span_N_m"], "N/m", flush=True)
            except Exception as exc:
                atomic_json(Path(cfg.get("_study_work_dir",paths["work"])) / "aero_summary.json", {"status": "failed", "error": str(exc)})
                print("[export] Aerodynamic export failed; completed temperature exports are retained.", flush=True)
                raise
        return result
    finally:
        solver.exit()


def export_cli(thermal=True, aero=True):
    import argparse
    from .config import load_config
    parser = argparse.ArgumentParser(description="Export solved wall/fluid temperatures and aerodynamic forces")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--inspect-fields", action="store_true")
    args = parser.parse_args()
    result = export_saved_case(load_config(args.config), thermal=thermal, aero=aero, inspect=args.inspect_fields)
    print("[export] Complete:", result)
