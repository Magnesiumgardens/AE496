"""Full 2-D internal-temperature screening using saved Mach-8 calibration."""
from __future__ import annotations

import json
import fnmatch
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .aerothermal_calibration import verify_surface_package
from .aerothermal_loading import read_load
from .config import load_config
from .internal_probes import axial_tip_probe, surface_normal_probes, prepare_probe_interpolation
from .internal_screening_metrics import first_threshold_crossing, screening_bar_outputs
from .peak_flux_reporting import _geometry_cases
from .reporting import _plotting, _save, load_yaml, write_csv
from .thermal import solve_thermal, save_results
from .thermal_loading import ThermalAnalysisError, positive
from .thermal_materials import from_material_config
from .utils import atomic_json, file_sha256

MODEL_VERSION = "1.1-startup-refined-internal-screen"


def native_launch_options(options, processors=None):
    """Use explicit shared-memory execution unless DMP is deliberately selected."""
    parallel = str(options.get("mapdl_parallel_mode", "smp")).strip().lower()
    if parallel not in {"smp", "dmp"}:
        raise ThermalAnalysisError("mapdl_parallel_mode must be smp or dmp")
    raw = options.get("processors", 4) if processors is None else processors
    if isinstance(raw, bool):
        raise ThermalAnalysisError("processors must be a positive integer")
    count = positive("processors", raw)
    if not count.is_integer():
        raise ThermalAnalysisError("processors must be a positive integer")
    return parallel, dict(nproc=int(count), additional_switches="-smp" if parallel == "smp" else "-dis")


def screening_settings(config_path, *, depth_mode=None, material_axes=None):
    path = Path(config_path).resolve()
    cfg = load_yaml(path)
    mode = depth_mode or cfg.get("depth_mode")
    axes = material_axes or cfg.get("material_axes_2d")
    if not isinstance(mode, str) or mode not in {"axial_tip", "surface_normal"}:
        raise ThermalAnalysisError("Specify depth_mode= axial_tip or surface_normal; the reference files do not define the 10 mm probe")
    if (not isinstance(axes, (list, tuple)) or len(axes) != 2
            or any(axis not in ("x", "y", "z") for axis in axes) or len(set(axes)) != 2):
        raise ThermalAnalysisError("Set material_axes_2d to two distinct material directions mapped to global X/Y, or supply --material-axes")
    if not isinstance(cfg.get("material_file"), str) or not cfg["material_file"].strip():
        raise ThermalAnalysisError("Set material_file to a thermal property YAML file")
    material_path = Path(cfg["material_file"])
    if not material_path.is_absolute():material_path = path.parent / material_path
    material_cfg = load_yaml(material_path)
    material_cfg["extrapolation"] = cfg.get("material_extrapolation", material_cfg.get("extrapolation", "error"))
    law = from_material_config(material_cfg)
    initial = positive("initial_temperature_k", cfg.get("initial_temperature_k", 300.))
    settings = {key: positive(key, cfg.get(name, default)) for key, name, default in (
        ("duration", "duration_s", 200.), ("dt0", "initial_time_step_s", .001),
        ("dtmin", "min_time_step_s", .000001), ("dtmax", "max_time_step_s", .5),
        ("depth_m", "depth_m", .01), ("limit_K", "limit_temperature_k", 1800.))}
    if not settings["dtmin"] <= settings["dt0"] <= settings["dtmax"] <= settings["duration"]:
        raise ThermalAnalysisError("Require minimum <= initial <= maximum time step <= duration")
    startup = float(cfg.get("startup_duration_s", 5.))
    if not np.isfinite(startup) or startup < 0:
        raise ThermalAnalysisError("startup_duration_s must be finite and nonnegative")
    settings["startup_duration"] = min(startup, settings["duration"])
    settings["startup_dtmax"] = positive("startup_max_time_step_s", cfg.get("startup_max_time_step_s", min(.05, settings["dtmax"])))
    if startup and not settings["dtmin"] <= settings["startup_dtmax"] <= settings["dtmax"]:
        raise ThermalAnalysisError("Startup maximum time step must lie between minimum and normal maximum")
    if not isinstance(cfg.get("require_min_surface_clearance", True), bool):
        raise ThermalAnalysisError("require_min_surface_clearance must be true or false")
    settings.update(initial_t=initial, rho=law.density, cp=float(law.specific_heat(initial)),
        k=float(law.conductivity(initial, axes[0])), material_axes_2d=list(axes), depth_mode=mode,
        require_min_surface_clearance=bool(cfg.get("require_min_surface_clearance", True)),
        include_rad=True, loading_mode="aerodynamic_surface_load")
    law.conductivity(initial, axes[1])  # Reject an absent material direction before launch.
    settings["mesh_size_m"] = positive("mesh_size_m", cfg.get("mesh_size_m", .0005))
    if cfg.get("reference_crossing_time_s") is not None:
        positive("reference_crossing_time_s", cfg["reference_crossing_time_s"])
        positive("reference_limit_temperature_k", cfg.get("reference_limit_temperature_k"))
    return cfg, material_cfg, law, settings, material_path.resolve()


def probe_screening(time, internal, definitions, limit):
    """Find crossings on each actual probe, avoiding interpolation of a moving maximum."""
    rows = [dict(first_threshold_crossing(time, internal[:, j], limit), probe_id=probe["probe_id"])
            for j, probe in enumerate(definitions)]
    worst = first_threshold_crossing(time, np.max(internal, axis=1), limit)
    reached = [row for row in rows if row["time_to_limit_s"] is not None]
    critical = min(reached, key=lambda row: row["time_to_limit_s"]) if reached else None
    # The envelope supplies sampled peak/final values only. A change of hottest
    # probe between output times cannot create an earlier synthetic crossing.
    for key in ("status", "time_to_limit_s", "lower_time_s", "upper_time_s",
                "lower_temperature_K", "upper_temperature_K", "crossing_method"):
        worst[key] = (critical or rows[0])[key]
    worst["critical_probe_id"] = critical["probe_id"] if critical else None
    worst["peak_probe_id"] = definitions[int(np.unravel_index(np.argmax(internal), internal.shape)[1])]["probe_id"]
    return worst, rows


def _signature(calibration, settings, material_cfg):
    # Geometry and CFD mesh hashes vary deliberately between noses. Flight,
    # gas/wall models, solid material, exposure and measurement must match.
    conditions = calibration["conditions"]
    return dict(conditions={key:conditions[key] for key in ("flight", "physics", "thermal")}, radiation=calibration["radiation"],
                rear_boundary=calibration.get("thermal_scope", {}).get("rear_boundary"),
                settings=settings, material=material_cfg)


def _case_inputs(case_config, screening_config, depth_mode=None, material_axes=None):
    cfg = load_config(case_config)
    if not np.isclose(float(cfg["flight"]["mach"]), 8., rtol=0, atol=1e-8):
        raise ThermalAnalysisError("This comparison uses existing Mach-8 calibration only; no flight sweep is performed")
    options, material_cfg, law, settings, material_path = screening_settings(screening_config,
        depth_mode=depth_mode, material_axes=material_axes)
    boundary, calibration = verify_surface_package(cfg)
    settings.update(emissivity=float(calibration["radiation"]["emissivity"]),
                    trad=float(calibration["radiation"]["external_radiation_temperature_K"]),
                    nose_direction=calibration["calibration_settings"]["nose_direction"])
    snapshot = dict(model_version=MODEL_VERSION, settings=settings, material=material_cfg,
        material_file_sha256=file_sha256(material_path), screening_config_sha256=file_sha256(screening_config),
        boundary_csv_sha256=file_sha256(boundary), source_sha256=calibration["source_sha256"],
        calibration_settings=calibration["calibration_settings"],
        source_geometry_sha256=calibration["source_geometry_sha256"],
        comparison_signature=_signature(calibration, settings, material_cfg))
    return cfg, options, law, material_path, boundary, snapshot


def _case_plots(output, time, internal, definitions, nnum, xyz, triangles, final, limit):
    """Render saved physical-time results; triangulation subdivision is for plotting only."""
    plt = _plotting()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time, internal, color="tab:blue", alpha=min(.4, 4 / internal.shape[1]), linewidth=.7)
    ax.plot(time, internal.max(axis=1), color="black", label="Maximum across internal probes")
    ax.axhline(limit, color="firebrick", linestyle="--", label=f"Screening limit {limit:g} K")
    ax.set(xlabel="Physical time [s]", ylabel="Internal probe temperature [K]", title="Solved internal temperature history")
    ax.grid(alpha=.3); ax.legend()
    _save(fig, output / "internal_temperature_history.png", plt)
    by_id = {int(node):i for i, node in enumerate(nnum)}
    indices = np.asarray([[by_id[int(node)] for node in cell] for cell in triangles])
    split = np.concatenate([indices[:, order] for order in ([0,3,5], [3,1,4], [5,4,2], [3,4,5])])
    import matplotlib.tri as mtri
    fig, ax = plt.subplots(figsize=(11, 5))
    field = ax.tripcolor(mtri.Triangulation(xyz[:, 0], xyz[:, 1], split), final, shading="gouraud", cmap="inferno")
    points = np.asarray([[p["x_m"], p["y_m"]] for p in definitions])
    ax.scatter(points[:, 0], points[:, 1], s=12, c="cyan", label="Internal probe locations")
    ax.set(xlabel="X [m]", ylabel="Y [m]", title=f"Final solid temperature at {time[-1]:g} s")
    ax.set_aspect("equal"); ax.legend(); fig.colorbar(field, ax=ax, label="Temperature [K]")
    _save(fig, output / "internal_temperature_final.png", plt)


def _probes(load, settings, direction):
    if settings["depth_mode"] == "surface_normal":
        result = surface_normal_probes(load.xy, load.segment_active, settings["depth_m"],
            require_min_clearance=settings["require_min_surface_clearance"])
        if not len(result["points"]):
            raise ThermalAnalysisError(f"No valid {settings['depth_m']*1000:g} mm surface-normal probes fit this solid; another heated surface may be closer. Do not substitute a nearest node")
        return result
    point = axial_tip_probe(load.xy, settings["depth_m"], direction)
    starts = load.xy[load.segment_active]
    edges = (np.roll(load.xy, -1, axis=0) - load.xy)[load.segment_active]
    fraction = np.clip(np.sum((point - starts) * edges, axis=1) / np.sum(edges**2, axis=1), 0, 1)
    clearance = float(np.min(np.linalg.norm(point - starts - fraction[:, None] * edges, axis=1)))
    return dict(points=np.asarray([point]), probes=[dict(probe_id="axial_tip", depth_m=settings["depth_m"],
        x_m=float(point[0]), y_m=float(point[1]), nearest_heated_surface_distance_m=clearance)], excluded=[])


def run_internal_case(case_config, screening_config, *, depth_mode=None, material_axes=None, processors=None):
    cfg, options, law, material_path, boundary, snapshot = _case_inputs(case_config, screening_config, depth_mode, material_axes)
    settings = snapshot["settings"]
    parallel, launch = native_launch_options(options, processors)
    load = read_load(boundary)
    load.material_law = law
    probes = _probes(load, settings, settings["nose_direction"])  # Validate geometry before using a license.
    work = Path(cfg["project"]["work_dir"])
    output = work / "internal_analysis"
    output.mkdir(parents=True, exist_ok=True)
    # Never mix a crashed run's partial files with a prior combined result.
    native = output / "native_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    native.mkdir(parents=True)
    receipt = dict(status="running", case=work.name, **snapshot, material_file=str(material_path),
        native_runtime=dict(parallel_mode=parallel, processors=launch["nproc"],
            additional_switches=launch["additional_switches"], native_directory=str(native),
            model_database=str(native / "internal_model.db"),
            result_output="Nodal temperatures (NSOL) at every solved substep; element outputs disabled"),
        depth_mode=settings["depth_mode"], depth_m=settings["depth_m"], limit_temperature_K=settings["limit_K"],
        probe_definitions=probes["probes"], excluded_probes=probes["excluded"],
        limitations=["The selected temperature is a screening limit, not a validated structural-failure model.",
            "The supplied 143 s value is a reference, not a calibration target.",
            "Material axes are fixed global directions; they do not automatically follow a curved fabric surface.",
            "Laminar planar Mach-8 CFD heating is one-way coupled; no oxidation, ablation, coating or trajectory is added.",
            "Emissivity is inherited from the existing CFD calibration, not inferred from the new material table."])
    atomic_json(output / "internal_summary.json", receipt)
    print(f"[internal] {work.name}: {settings['depth_mode']}, depth={settings['depth_m']*1000:g} mm, material={law.name}", flush=True)
    print(f"[internal] MAPDL mode={parallel}, processors={launch['nproc']}; native files: {native}", flush=True)
    engine = None
    try:
        from ansys.mapdl.core import launch_mapdl
        engine = launch_mapdl(version=int(options.get("mapdl_version", 251)), mode="grpc",
            **launch, run_location=str(native), jobname="internal_nose",
            start_timeout=positive("mapdl_start_timeout_s", options.get("mapdl_start_timeout_s", 120)),
            log_apdl=str(native / "mapdl_commands.log"), mapdl_output=str(native / "mapdl_console.out"))
        mesh_size = settings["mesh_size_m"]
        nnum, xyz, values, times, elements = solve_thermal(engine, load, settings, mesh_size,
            save_nodal_temperatures_only=True, model_database_file="internal_model")
        temperatures = np.vstack([np.full(len(nnum), settings["initial_t"]), *values])
        time = np.r_[0., np.asarray(times)]
        np.savez_compressed(output / "internal_solution.npz", node_ids=nnum, coordinates_m=xyz,
            triangle_node_ids=load.solid_triangles, time_s=time, temperatures_K=temperatures)
        receipt["material_range"] = law.range_diagnostics(temperatures)
        if law.extrapolation == "error" and receipt["material_range"]["status"] == "outside_range":
            raise ThermalAnalysisError("Solved temperatures exceed the supplied material table; raw solution retained. Provide data or explicitly select endpoint extrapolation")
        save_results(output, load, nnum, xyz, values, times, settings, mesh_size)
        receipt["energy_balance"] = load.energy_balance
        if load.energy_balance["status"] != "passed":
            balance = load.energy_balance
            raise ThermalAnalysisError(f"Physical enthalpy energy check failed: worst step at {balance['worst_step_time_s']:g} s, "
                f"error/tolerance={balance['maximum_step_error_to_tolerance']:.4g}; "
                f"final balance passed={balance['final_balance_passed']}, signs passed={balance['energy_sign_passed']}. "
                f"Saved results retained. Inspect {output / 'energy_balance.json'} before using the comparison")
        interpolation = prepare_probe_interpolation(nnum, xyz, load.solid_triangles, probes["points"])
        internal = interpolation.evaluate(temperatures)
        history = [dict(design=work.name, time_s=float(t), probe_id=probe["probe_id"],
            temperature_K=float(internal[i, j]), **{k:v for k,v in probe.items() if k != "probe_id"})
            for i, t in enumerate(time) for j, probe in enumerate(probes["probes"])]
        write_csv(output / "internal_temperature_history.csv", history)
        worst = np.max(internal, axis=1)
        metrics, probe_metrics = probe_screening(time, internal, probes["probes"], settings["limit_K"])
        metrics.update(design=work.name, depth_m=settings["depth_m"], depth_mode=settings["depth_mode"],
            material_extrapolation_used=receipt["material_range"]["status"] == "outside_range")
        write_csv(output / "internal_probe_screening.csv", probe_metrics)
        write_csv(output / "internal_worst_temperature_history.csv", [dict(design=work.name, time_s=float(t),
            temperature_K=float(temperature)) for t, temperature in zip(time, worst)])
        _case_plots(output, time, internal, probes["probes"], nnum, xyz, load.solid_triangles,
                    temperatures[-1], settings["limit_K"])
        receipt.update(status="complete", screening=metrics, material_axis_mapping=load.native_material,
            radiation_boundary=load.radiation_boundary, probe_count=len(probes["probes"]),
            interpretation="Worst solved probe temperature at each saved time; crossing computed separately per probe, then the earliest is selected",
            result_sha256={p.name:file_sha256(p) for p in output.iterdir() if p.suffix in {".csv", ".npz"} or p.name == "energy_balance.json"})
        atomic_json(output / "internal_summary.json", receipt)
        print(f"[internal] {work.name}: {metrics['status']}; peak={metrics['peak_temperature_K']:g} K", flush=True)
        return receipt
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        atomic_json(output / "internal_summary.json", receipt)
        print(f"[internal] Native diagnostics retained: {native}", flush=True)
        raise
    finally:
        if engine is not None:
            try:
                engine.exit()
            except Exception:
                if receipt["status"] != "failed":raise


def compare_internal(runs_dir, screening_config, *, geometry_dir=None, cases=None, output_dir=None,
                     depth_mode=None, material_axes=None, report_only=False, processors=None):
    # Required physical choices are checked even for report regeneration.
    options, _, _, settings, _ = screening_settings(screening_config, depth_mode=depth_mode, material_axes=material_axes)
    native_launch_options(options, processors)  # Validate before starting any case.
    runs = Path(runs_dir).resolve()
    if not runs.is_dir():raise ThermalAnalysisError(f"Runs directory does not exist: {runs}")
    if geometry_dir is not None or (runs.parent / "geometry_batch").is_dir():
        candidates, inventory, geometry = _geometry_cases(runs, geometry_dir, "*.x_t", cases)
    else:
        patterns = [pattern.casefold() for pattern in cases] if cases else ["*"]
        candidates = [work for work in sorted(runs.iterdir()) if (work / "_batch/config.yaml").is_file()
                      and any(fnmatch.fnmatchcase(work.name.casefold(), p) for p in patterns)]
        inventory, geometry = [], None
    if not candidates:
        raise ThermalAnalysisError("No case folders selected; check --runs-dir, --geometry-dir and --cases")
    selected_cad = {item["case"].casefold():item["geometry_file"] for item in inventory if item["status"] != "not_selected"}
    output = Path(output_dir).resolve() if output_dir else runs / "comparison/internal_heating"
    output.mkdir(parents=True, exist_ok=True)
    # Remove only this command's old charts/CSVs before regenerating. An empty
    # or rejected selection must not leave a previous ranking visible as current.
    for name in ("peak_internal_temperature_comparison", "time_to_internal_limit_comparison"):
        for suffix in (".csv", ".png"):(output / (name + suffix)).unlink(missing_ok=True)
    (output / "internal_temperature_history_comparison.png").unlink(missing_ok=True)
    rows, histories, excluded, signature = [], {}, [], None
    for work in candidates:
        try:
            case_cfg, _, _, _, _, current = _case_inputs(work / "_batch/config.yaml", screening_config, depth_mode, material_axes)
            if case_cfg is not None and Path(case_cfg["project"]["work_dir"]).resolve() != work.resolve():
                raise ThermalAnalysisError("Case configuration work_dir points at a different nose folder")
            if work.name.casefold() in selected_cad and file_sha256(selected_cad[work.name.casefold()]) != current["source_geometry_sha256"]:
                raise ThermalAnalysisError("Selected batch CAD differs from this case's calibrated geometry")
            if signature is not None and current["comparison_signature"] != signature:
                raise ThermalAnalysisError("Flight, wall/radiation, material or internal-screening assumptions differ from the first included case")
            if report_only:
                receipt = json.loads((work / "internal_analysis/internal_summary.json").read_text())
                if receipt.get("status") != "complete" or any(receipt.get(key) != value for key, value in current.items()):
                    raise ThermalAnalysisError("Saved internal result is incomplete or its model, material, configuration or CFD calibration changed")
                required = {"internal_solution.npz", "internal_temperature_history.csv", "internal_probe_screening.csv",
                            "internal_worst_temperature_history.csv", "solid_energy_history.csv"}
                if not required <= set(receipt.get("result_sha256", {})):
                    raise ThermalAnalysisError("Saved internal result receipt lacks required physical-time outputs")
                for name, digest in receipt.get("result_sha256", {}).items():
                    if file_sha256(work / "internal_analysis" / name) != digest:raise ThermalAnalysisError("Saved internal result hash changed")
                with np.load(work / "internal_analysis/internal_solution.npz", allow_pickle=False) as saved:
                    definitions = receipt["probe_definitions"]
                    interpolation = prepare_probe_interpolation(saved["node_ids"], saved["coordinates_m"],
                        saved["triangle_node_ids"], [[p["x_m"], p["y_m"]] for p in definitions])
                    internal = interpolation.evaluate(saved["temperatures_K"])
                    _case_plots(work / "internal_analysis", saved["time_s"], internal, definitions, saved["node_ids"],
                        saved["coordinates_m"], saved["triangle_node_ids"], saved["temperatures_K"][-1], settings["limit_K"])
            else:
                receipt = run_internal_case(work / "_batch/config.yaml", screening_config,
                    depth_mode=depth_mode, material_axes=material_axes, processors=processors)
            if receipt.get("energy_balance", {}).get("status") != "passed":raise ThermalAnalysisError("Internal energy check is not passed")
            from .reporting import read_csv
            history = read_csv(work / "internal_analysis/internal_worst_temperature_history.csv")
            if not history:raise ThermalAnalysisError("Saved internal history is empty")
            rows.append(receipt["screening"])
            signature = receipt["comparison_signature"]
            histories[work.name] = history
        except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as exc:
            excluded.append(dict(case=work.name, reason=str(exc)))
            print(f"[internal] {work.name}: excluded: {exc}", flush=True)
    reference_time = options.get("reference_crossing_time_s")
    if reference_time is not None and not np.isclose(settings["limit_K"], options["reference_limit_temperature_k"], rtol=0, atol=1e-8):
        reference_time = None
    files = screening_bar_outputs(rows, output, limit_K=settings["limit_K"], reference_time_s=reference_time) if rows else {}
    if histories:
        plt = _plotting()
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, history in histories.items():
            ax.plot([float(row["time_s"]) for row in history], [float(row["temperature_K"]) for row in history], label=name)
        ax.axhline(settings["limit_K"], color="black", linestyle="--", label=f"Screening limit {settings['limit_K']:g} K")
        ax.set(xlabel="Physical time [s]", ylabel="Worst internal probe temperature [K]",
               title=f"Internal temperature at {settings['depth_m']*1000:g} mm ({settings['depth_mode']})")
        ax.grid(alpha=.3); ax.legend()
        files["internal_temperature_history_comparison_png"] = _save(fig, output / "internal_temperature_history_comparison.png", plt)
    result = dict(status="complete" if rows and not excluded else "partial" if rows else "unavailable",
        case_count=len(rows), rows=rows, excluded=excluded, files=files,
        geometry_directory=geometry, geometry_inventory=inventory,
        comparison_signature=signature,
        reference=dict(time_s=options.get("reference_crossing_time_s"), limit_temperature_K=options.get("reference_limit_temperature_k"),
            description=options.get("reference_description"), time_line_displayed=reference_time is not None,
            interpretation="User-supplied reference only; its flight history differs and is not independently reproduced"),
        interpretation="Threshold-crossing screen at explicitly defined internal probes; no claim of structural failure or whole-vehicle limiting component")
    atomic_json(output / "internal_comparison.json", result)
    print(f"[internal] Comparison: {output}", flush=True)
    return result
