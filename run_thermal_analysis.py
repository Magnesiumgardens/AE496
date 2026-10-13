from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import math
import re
import shutil
import sys

import numpy as np
import yaml

from pipeline.trade_study import THERMAL_MODEL_VERSION, thermal_screening_metrics


class ThermalAnalysisError(RuntimeError):
    pass

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run transient thermal analysis for a completed CFD case."
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the case run directory, e.g. runs/blunt_1",
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="CFD YAML configuration file.",
    )

    parser.add_argument(
        "--thermal-config",
        default="thermal_analysis.yaml",
        help="Thermal-analysis YAML configuration file.",
    )

    return parser.parse_args()

def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise ThermalAnalysisError(f"Missing configuration file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ThermalAnalysisError(f"{path} must contain a YAML mapping.")
    return data


def positive(name: str, value) -> float:
    if value is None:
        raise ThermalAnalysisError(
            f"thermal_analysis.yaml: '{name}' is required and cannot be null."
        )
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ThermalAnalysisError(f"'{name}' must be numeric; got {value!r}.") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ThermalAnalysisError(f"'{name}' must be > 0; got {value}.")
    return value


def get_work_dir(cfd_cfg: dict) -> Path:
    return Path(cfd_cfg.get("project", {}).get("work_dir", "run_dir")).resolve()


def get_bc_path(cfd_cfg: dict, ta_cfg: dict) -> Path:
    override = ta_cfg.get("boundary_csv")
    if override:
        return Path(override).resolve()
    bc_dir = cfd_cfg.get("thermal_export", {}).get("output_directory", "thermal_bc")
    return (get_work_dir(cfd_cfg) / bc_dir / "body_wall_faces.csv").resolve()


def read_boundary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise ThermalAnalysisError(
            f"Thermal BC file not found: {path}\n"
            "Run export_thermal_only.py first."
        )

    required = {
        "surface_order_index",
        "x_m",
        "y_m",
        "heat_flux_into_wall_W_m2",
    }
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ThermalAnalysisError(
                f"{path} is missing columns: {sorted(missing)}. "
                f"Available: {sorted(fields)}"
            )
        for row in reader:
            try:
                rows.append(
                    (
                        int(float(row["surface_order_index"])),
                        float(row["x_m"]),
                        float(row["y_m"]),
                        float(row["heat_flux_into_wall_W_m2"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ThermalAnalysisError(f"Bad numeric row in {path}: {row}") from exc

    rows.sort(key=lambda r: r[0])
    if len(rows) < 3:
        raise ThermalAnalysisError("Need at least 3 thermal wall points.")

    xy = np.asarray([[r[1], r[2]] for r in rows], dtype=float)
    q = np.asarray([r[3] for r in rows], dtype=float)

    if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(q)):
        raise ThermalAnalysisError("Thermal BC CSV contains NaN/Inf values.")

    # Remove consecutive duplicate points.
    body_span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1.0)
    tol = max(1e-12, body_span * 1e-10)
    keep = [0]
    for i in range(1, len(xy)):
        if np.linalg.norm(xy[i] - xy[keep[-1]]) > tol:
            keep.append(i)
    if len(keep) > 2 and np.linalg.norm(xy[keep[-1]] - xy[keep[0]]) <= tol:
        keep.pop()
    xy = xy[keep]
    q = q[keep]

    if len(xy) < 3:
        raise ThermalAnalysisError("Too few unique wall points after duplicate removal.")

    # Force counter-clockwise order for a clean 2-D area.
    area2 = float(
        np.dot(xy[:, 0], np.roll(xy[:, 1], -1))
        - np.dot(xy[:, 1], np.roll(xy[:, 0], -1))
    )
    if abs(area2) <= 1e-16:
        raise ThermalAnalysisError("CFD wall points do not enclose a nonzero 2-D area.")
    if area2 < 0:
        xy = xy[::-1].copy()
        q = q[::-1].copy()

    return xy, q


def auto_mesh_size(xy: np.ndarray, configured) -> float:
    if configured is not None:
        return positive("mesh_size_m", configured)
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    ds = ds[ds > 0]
    if len(ds) == 0:
        raise ThermalAnalysisError("Could not infer mesh size from the wall coordinates.")
    body_span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])))
    return min(max(2.0 * float(np.median(ds)), body_span / 500.0), body_span / 40.0)


def validate_settings(cfd_cfg: dict, ta_cfg: dict) -> dict:
    mode = ta_cfg.get("loading", {}).get("mode", "fixed_heat_flux")
    if mode != "fixed_heat_flux":
        raise ThermalAnalysisError(
            f"loading.mode={mode!r} is unsupported by this uploaded thermal runner. "
            "It applies fixed HFLUX and does not read the adiabatic endpoint. "
            "Provide the endpoint-convection runner, or explicitly select fixed_heat_flux "
            "for a labeled legacy screening study."
        )
    material = ta_cfg.get("material", {})
    transient = ta_cfg.get("transient", {})
    cfd_thermal = cfd_cfg.get("thermal", {})

    k = positive("material.conductivity_w_mk", material.get("conductivity_w_mk"))
    rho = positive("material.density_kg_m3", material.get("density_kg_m3"))
    cp = positive("material.specific_heat_j_kgk", material.get("specific_heat_j_kgk"))
    duration = positive("transient.duration_s", transient.get("duration_s"))

    initial_t = ta_cfg.get("initial_temperature_k")
    if initial_t is None:
        initial_t = cfd_thermal.get("wall_temperature_k", 300.0)
    initial_t = positive("initial_temperature_k", initial_t)

    dt0 = transient.get("initial_time_step_s")
    dt0 = positive("transient.initial_time_step_s", dt0 if dt0 is not None else duration / 100.0)
    dtmin = transient.get("min_time_step_s")
    dtmin = positive("transient.min_time_step_s", dtmin if dtmin is not None else dt0 / 10.0)
    dtmax = transient.get("max_time_step_s")
    dtmax = positive("transient.max_time_step_s", dtmax if dtmax is not None else min(duration / 10.0, dt0 * 10.0))
    dtmax = min(dtmax, duration)
    if not (dtmin <= dt0 <= dtmax):
        raise ThermalAnalysisError(
            "Require min_time_step_s <= initial_time_step_s <= max_time_step_s."
        )

    include_rad = bool(ta_cfg.get("include_radiation", True))
    emissivity = ta_cfg.get("emissivity")
    if emissivity is None:
        emissivity = cfd_thermal.get("emissivity", 0.90)
    emissivity = float(emissivity)
    if not 0.0 <= emissivity <= 1.0:
        raise ThermalAnalysisError("emissivity must be between 0 and 1.")

    trad = ta_cfg.get("radiation_temperature_k")
    if trad is None:
        trad = cfd_thermal.get("external_radiation_temperature_k", 3.0)
    trad = positive("radiation_temperature_k", trad)

    return {
        "material_name": str(material.get("name", "user_material")),
        "k": k,
        "rho": rho,
        "cp": cp,
        "duration": duration,
        "initial_t": initial_t,
        "dt0": dt0,
        "dtmin": dtmin,
        "dtmax": dtmax,
        "include_rad": include_rad,
        "emissivity": emissivity,
        "trad": trad,
    }


def clean_temp(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    # With Kelvin initial temperature > 0, a returned zero is an invalid value.
    out[out <= 0.0] = np.nan
    return out


def temp_stats(values: np.ndarray) -> dict:
    good = values[np.isfinite(values)]
    if len(good) == 0:
        raise ThermalAnalysisError("MAPDL returned no valid nodal temperatures.")
    return {
        "minimum_K": float(np.min(good)),
        "mean_K": float(np.mean(good)),
        "maximum_K": float(np.max(good)),
    }


def write_nodes(path: Path, nnum, xyz, temp) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "x_m", "y_m", "z_m", "temperature_K"])
        for nid, p, t in zip(nnum, xyz, temp):
            w.writerow([int(nid), float(p[0]), float(p[1]), float(p[2]), "" if not np.isfinite(t) else float(t)])


def write_wall(path: Path, bc_xy, bc_q, nnum, xyz, temp) -> None:
    node_xy = xyz[:, :2]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "surface_order_index", "x_m", "y_m",
            "applied_heat_flux_into_wall_W_m2",
            "nearest_thermal_node_id", "nearest_thermal_node_distance_m",
            "thermal_wall_temperature_K",
        ])
        for i, (p, q) in enumerate(zip(bc_xy, bc_q)):
            d2 = np.sum((node_xy - p) ** 2, axis=1)
            j = int(np.argmin(d2))
            t = float(temp[j])
            w.writerow([
                i, float(p[0]), float(p[1]), float(q), int(nnum[j]),
                float(np.sqrt(d2[j])), "" if not math.isfinite(t) else t,
            ])


def get_visualization_settings(ta_cfg: dict) -> dict:
    cfg = ta_cfg.get("visualization", {}) or {}

    max_frames = int(cfg.get("max_animation_frames", 60))
    if max_frames < 2:
        max_frames = 2

    fps = float(cfg.get("fps", 8))
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 8.0

    dpi = int(cfg.get("dpi", 160))
    if dpi < 72:
        dpi = 72

    return {
        "enabled": bool(cfg.get("enabled", True)),
        "save_final_temperature_image": bool(
            cfg.get("save_final_temperature_image", True)
        ),
        "save_heat_flux_image": bool(
            cfg.get("save_heat_flux_image", True)
        ),
        "save_temperature_history_image": bool(
            cfg.get("save_temperature_history_image", True)
        ),
        "make_gif": bool(cfg.get("make_gif", True)),
        "save_frame_pngs": bool(cfg.get("save_frame_pngs", False)),
        "max_animation_frames": max_frames,
        "fps": fps,
        "dpi": dpi,
    }


def choose_snapshot_indices(nsets: int, max_frames: int) -> set[int]:
    """Return 1-based MAPDL result-set numbers selected for visualization."""
    if nsets <= 0:
        return set()
    if nsets <= max_frames:
        return set(range(1, nsets + 1))

    selected = np.linspace(
        1,
        nsets,
        num=max_frames,
        dtype=int,
    )
    return set(int(v) for v in selected)


def _masked_triangulation(xyz: np.ndarray, boundary_xy: np.ndarray):
    """
    Create a 2-D plotting triangulation and mask triangles outside the body.

    MAPDL's FE mesh remains authoritative for the analysis. This
    triangulation is only for PDR visualization.
    """
    import matplotlib.tri as mtri
    from matplotlib.path import Path as MplPath

    x = np.asarray(xyz[:, 0], dtype=float)
    y = np.asarray(xyz[:, 1], dtype=float)

    tri = mtri.Triangulation(x, y)

    triangles = tri.triangles
    cx = np.mean(x[triangles], axis=1)
    cy = np.mean(y[triangles], axis=1)
    centroids = np.column_stack([cx, cy])

    body_path = MplPath(
        np.vstack([boundary_xy, boundary_xy[0]])
    )
    inside = body_path.contains_points(centroids)

    if np.any(inside):
        tri.set_mask(~inside)

    return tri


def _temperature_limits(snapshots: list[dict]) -> tuple[float, float]:
    values = []
    for snap in snapshots:
        t = np.asarray(snap["temperature_K"], dtype=float)
        t = t[np.isfinite(t)]
        if len(t):
            values.append(t)

    if not values:
        raise ThermalAnalysisError(
            "No valid temperature values were available for visualization."
        )

    all_t = np.concatenate(values)
    tmin = float(np.min(all_t))
    tmax = float(np.max(all_t))

    if math.isclose(tmin, tmax, rel_tol=1e-12, abs_tol=1e-9):
        pad = max(1.0, abs(tmin) * 0.01)
        tmin -= pad
        tmax += pad

    return tmin, tmax


def create_visualizations(
    output_dir: Path,
    boundary_xy: np.ndarray,
    boundary_q: np.ndarray,
    xyz: np.ndarray,
    history: list[dict],
    snapshots: list[dict],
    viz_cfg: dict,
) -> dict:
    """
    Generate PDR-friendly 2-D still images and an optional heating GIF.

    main() treats plotting as non-fatal, so a completed MAPDL thermal
    solution is never discarded because visualization fails.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ThermalAnalysisError(
            "Visualization is enabled but matplotlib is not installed. "
            "Install it with: "
            r".\.venv\Scripts\python.exe -m pip install matplotlib pillow"
        ) from exc

    viz_dir = output_dir / "visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    tri = _masked_triangulation(xyz, boundary_xy)

    if not snapshots:
        raise ThermalAnalysisError(
            "No thermal snapshots were available for visualization."
        )

    final_temp = np.asarray(
        snapshots[-1]["temperature_K"],
        dtype=float,
    )
    final_time = float(snapshots[-1]["time_s"])

    tmin, tmax = _temperature_limits(snapshots)
    levels = np.linspace(tmin, tmax, 24)

    if viz_cfg["save_final_temperature_image"]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        contour = ax.tricontourf(
            tri,
            final_temp,
            levels=levels,
        )
        fig.colorbar(
            contour,
            ax=ax,
            label="Temperature [K]",
        )
        ax.plot(
            np.r_[boundary_xy[:, 0], boundary_xy[0, 0]],
            np.r_[boundary_xy[:, 1], boundary_xy[0, 1]],
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(
            "Final 2-D Temperature Field\n"
            f"t = {final_time:.4g} s, "
            f"Tmax = {np.nanmax(final_temp):.1f} K"
        )
        fig.tight_layout()
        path = viz_dir / "final_temperature_field.png"
        fig.savefig(path, dpi=viz_cfg["dpi"])
        plt.close(fig)
        outputs["final_temperature_field_png"] = str(path)

    if viz_cfg["save_heat_flux_image"]:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            boundary_xy[:, 0],
            boundary_xy[:, 1],
            c=boundary_q,
            s=18,
        )
        fig.colorbar(
            scatter,
            ax=ax,
            label="Aerothermal heat flux into wall [W/m²]",
        )
        ax.plot(
            np.r_[boundary_xy[:, 0], boundary_xy[0, 0]],
            np.r_[boundary_xy[:, 1], boundary_xy[0, 1]],
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Applied Aerothermal Heat-Flux Distribution")
        fig.tight_layout()
        path = viz_dir / "applied_heat_flux_map.png"
        fig.savefig(path, dpi=viz_cfg["dpi"])
        plt.close(fig)
        outputs["applied_heat_flux_map_png"] = str(path)

    if (
        viz_cfg["save_temperature_history_image"]
        and history
    ):
        time_s = np.asarray(
            [row["time_s"] for row in history],
            dtype=float,
        )
        tlo = np.asarray(
            [row["minimum_K"] for row in history],
            dtype=float,
        )
        tmean = np.asarray(
            [row["mean_K"] for row in history],
            dtype=float,
        )
        thi = np.asarray(
            [row["maximum_K"] for row in history],
            dtype=float,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(time_s, tlo, label="Minimum")
        ax.plot(time_s, tmean, label="Mean")
        ax.plot(time_s, thi, label="Maximum")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("Thermal Response History")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        path = viz_dir / "temperature_history.png"
        fig.savefig(path, dpi=viz_cfg["dpi"])
        plt.close(fig)
        outputs["temperature_history_png"] = str(path)

    if viz_cfg["make_gif"] and len(snapshots) >= 2:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ThermalAnalysisError(
                "GIF output is enabled but Pillow is not installed. "
                "Install it with: "
                r".\.venv\Scripts\python.exe -m pip install pillow"
            ) from exc

        frame_dir = viz_dir / "animation_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []

        for frame_number, snap in enumerate(snapshots):
            temp = np.asarray(
                snap["temperature_K"],
                dtype=float,
            )
            time_s = float(snap["time_s"])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            contour = ax.tricontourf(
                tri,
                temp,
                levels=levels,
            )
            fig.colorbar(
                contour,
                ax=ax,
                label="Temperature [K]",
            )
            ax.plot(
                np.r_[boundary_xy[:, 0], boundary_xy[0, 0]],
                np.r_[boundary_xy[:, 1], boundary_xy[0, 1]],
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(
                "2-D Aerothermal Heating Response\n"
                f"t = {time_s:.4g} s, "
                f"Tmax = {np.nanmax(temp):.1f} K"
            )
            fig.tight_layout()

            frame_path = (
                frame_dir
                / f"frame_{frame_number:04d}.png"
            )
            fig.savefig(
                frame_path,
                dpi=viz_cfg["dpi"],
            )
            plt.close(fig)
            frame_paths.append(frame_path)

        images = []
        try:
            for frame_path in frame_paths:
                with Image.open(frame_path) as im:
                    images.append(im.convert("RGB").copy())

            gif_path = viz_dir / "nose_heating_animation.gif"
            frame_duration_ms = max(
                20,
                int(round(1000.0 / viz_cfg["fps"])),
            )
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=frame_duration_ms,
                loop=0,
            )
            outputs["nose_heating_animation_gif"] = str(
                gif_path
            )
        finally:
            for im in images:
                try:
                    im.close()
                except Exception:
                    pass

        if viz_cfg["save_frame_pngs"]:
            outputs["animation_frames_directory"] = str(
                frame_dir
            )
        else:
            shutil.rmtree(
                frame_dir,
                ignore_errors=True,
            )

    return outputs


def _resolve_from_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "case"

def run_thermal_analysis(
    run_dir: str | Path,
    cfd_config: str | Path = "config.yaml",
    thermal_config: str | Path = "thermal_analysis.yaml",
) -> dict:
    root = Path(__file__).resolve().parent
    run_dir = Path(run_dir).resolve()

    cfd_config_path = _resolve_from_root(root, cfd_config)
    thermal_config_path = _resolve_from_root(root, thermal_config)

    cfd_cfg = load_yaml(cfd_config_path)
    ta_cfg = load_yaml(thermal_config_path)
    settings = validate_settings(cfd_cfg, ta_cfg)

    # Use the thermal BC belonging to the explicitly selected case.
    bc_path = run_dir / "thermal_bc" / "body_wall_faces.csv"

    if not bc_path.exists():
        raise ThermalAnalysisError(
            f"Thermal BC file not found:\n{bc_path}"
        )

    xy, q = read_boundary(bc_path)
    mesh_size = auto_mesh_size(
        xy,
        ta_cfg.get("mesh_size_m"),
    )

    # Thermal results belong inside the selected case directory.
    output_dir = run_dir / "thermal_analysis"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    project_name = _safe_name(run_dir.name)

    # Keep MAPDL scratch/log files separate from the selected case path.
    mapdl_run_dir = (
        root
        / "logs"
        / "thermal_mapdl"
        / project_name
    )
    mapdl_run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[thermal] Case directory: {run_dir}")
    print(f"[thermal] BC file: {bc_path}")
    print(f"[thermal] Wall points: {len(xy)}")
    print(f"[thermal] Material: {settings['material_name']}")
    print(f"[thermal] Mesh size: {mesh_size:.6e} m")
    print(f"[thermal] Duration: {settings['duration']:.6g} s")
    print(f"[thermal] Output: {output_dir}")


    try:
        from ansys.mapdl.core import launch_mapdl
    except ImportError as exc:
        raise ThermalAnalysisError(
            "PyMAPDL is not installed. Install it with:\n"
            r"  .\.venv\Scripts\python.exe -m pip install ansys-mapdl-core"
        ) from exc

    mapdl = None
    try:
        mapdl = launch_mapdl(
            version=ta_cfg.get("mapdl_version", 251),
            run_location=str(mapdl_run_dir),
            jobname="thermal2d",
            nproc=int(ta_cfg.get("processors", cfd_cfg.get("runtime", {}).get("processors", 4))),
            cleanup_on_exit=False,
            remove_temp_dir_on_exit=False,
            start_timeout=120,
        )

        mapdl.clear()
        mapdl.prep7()
        mapdl.et(1, "PLANE293")
        mapdl.keyopt(1, 1, 1)  # 6-node triangular PLANE293
        mapdl.mp("KXX", 1, settings["k"])
        mapdl.mp("KYY", 1, settings["k"])
        mapdl.mp("DENS", 1, settings["rho"])
        mapdl.mp("C", 1, settings["cp"])

        keypoints = [int(mapdl.k("", float(x), float(y), 0.0)) for x, y in xy]
        lines = []
        for i in range(len(keypoints)):
            lines.append(int(mapdl.l(keypoints[i], keypoints[(i + 1) % len(keypoints)])))

        mapdl.allsel()
        area_num = int(mapdl.al("ALL"))
        if area_num <= 0:
            raise ThermalAnalysisError("MAPDL could not create the 2-D body area.")

        # HFLUX on PLANE293 is positive INTO the element. This directly matches
        # heat_flux_into_wall_W_m2 from the CFD exporter.
        for i, line_num in enumerate(lines):
            mapdl.sfl(line_num, "HFLUX", float(q[i]), float(q[(i + 1) % len(q)]))
            if settings["include_rad"]:
                mapdl.sfl(
                    line_num,
                    "RAD",
                    settings["emissivity"],
                    settings["emissivity"],
                    settings["trad"],
                    settings["trad"],
                )

        mapdl.toffst(0.0)  # Kelvin is already absolute.
        if settings["include_rad"]:
            mapdl.stef(5.670374419e-8)

        mapdl.esize(mesh_size)
        mapdl.mshape(1, "2D")
        mapdl.mshkey(0)
        mapdl.amesh(area_num)
        mapdl.allsel()
        mapdl.sftran()

        n_nodes = len(mapdl.mesh.nnum)
        n_elem = len(mapdl.mesh.enum)
        if n_nodes < 3 or n_elem < 1:
            raise ThermalAnalysisError(f"Invalid thermal mesh: {n_nodes} nodes, {n_elem} elements.")
        print(f"[thermal] Mesh: {n_nodes} nodes, {n_elem} elements")

        mapdl.finish()
        mapdl.slashsolu()
        mapdl.antype("TRANS")
        mapdl.trnopt("FULL")
        mapdl.timint("ON", "THERM")
        mapdl.tunif(settings["initial_t"])
        mapdl.kbc(1)  # step the constant aero-heating load on at t=0
        mapdl.autots("ON")
        mapdl.time(settings["duration"])
        mapdl.deltim(settings["dt0"], settings["dtmin"], settings["dtmax"])
        mapdl.outres("ALL", "ALL")
        mapdl.neqit(100)

        print("[thermal] Solving transient thermal model...")
        solve_text = str(mapdl.solve())
        if "*** ERROR ***" in solve_text.upper():
            raise ThermalAnalysisError("MAPDL solve error:\n" + solve_text[-5000:])

        mapdl.finish()
        mapdl.post1()
        mapdl.set("LAST")

        nnum = np.asarray(mapdl.mesh.nnum, dtype=int)
        xyz = np.asarray(mapdl.mesh.nodes, dtype=float)
        temp_final = clean_temp(np.asarray(mapdl.post_processing.nodal_temperature(), dtype=float))
        if not (len(nnum) == len(xyz) == len(temp_final)):
            raise ThermalAnalysisError(
                f"Result size mismatch: nnum={len(nnum)}, xyz={len(xyz)}, temp={len(temp_final)}"
            )

        final_stats = temp_stats(temp_final)
        nodes_csv = output_dir / "thermal_nodes_final.csv"
        wall_csv = output_dir / "wall_temperature_final.csv"
        hist_csv = output_dir / "temperature_history.csv"
        wall_hist_csv = output_dir / "wall_temperature_history.csv"
        summary_json = output_dir / "thermal_summary.json"

        write_nodes(nodes_csv, nnum, xyz, temp_final)
        write_wall(wall_csv, xy, q, nnum, xyz, temp_final)

        # Precompute the nearest thermal node for each ordered CFD wall point once.
        # The thermal mesh is static, so the same mapping is valid for every result set.
        node_xy = xyz[:, :2]
        wall_node_index = []
        wall_node_distance = []
        for p in xy:
            d2 = np.sum((node_xy - p) ** 2, axis=1)
            j = int(np.argmin(d2))
            wall_node_index.append(j)
            wall_node_distance.append(float(np.sqrt(d2[j])))

        history = []
        wall_history = []
        nsets = int(mapdl.post_processing.nsets)
        times = list(mapdl.post_processing.time_values)

        viz_cfg = get_visualization_settings(ta_cfg)
        selected_sets = (
            choose_snapshot_indices(
                nsets,
                viz_cfg["max_animation_frames"],
            )
            if viz_cfg["enabled"]
            else set()
        )
        snapshots = []

        for i in range(1, nsets + 1):
            mapdl.set(1, i)
            ti = clean_temp(
                np.asarray(
                    mapdl.post_processing.nodal_temperature(),
                    dtype=float,
                )
            )

            if len(ti) != len(xyz):
                raise ThermalAnalysisError(
                    f"Result set {i} returned {len(ti)} nodal temperatures "
                    f"for a mesh containing {len(xyz)} nodes."
                )

            row = temp_stats(ti)
            row["time_s"] = (
                float(times[i - 1])
                if i - 1 < len(times)
                else float("nan")
            )
            history.append(row)

            # Preserve the full wall-temperature field at every saved time.
            # This is the richer output used by compare_wall_temperatures.py.
            for surface_index, (p, q_wall, j, distance) in enumerate(
                zip(xy, q, wall_node_index, wall_node_distance)
            ):
                wall_t = float(ti[j])
                wall_history.append(
                    {
                        "time_s": row["time_s"],
                        "surface_order_index": surface_index,
                        "x_m": float(p[0]),
                        "y_m": float(p[1]),
                        "applied_heat_flux_into_wall_W_m2": float(q_wall),
                        "nearest_thermal_node_id": int(nnum[j]),
                        "nearest_thermal_node_distance_m": float(distance),
                        "wall_temperature_K": (
                            wall_t if math.isfinite(wall_t) else float("nan")
                        ),
                    }
                )

            if i in selected_sets:
                snapshots.append(
                    {
                        "result_set": i,
                        "time_s": row["time_s"],
                        "temperature_K": ti.copy(),
                    }
                )

        # Guarantee that the actual final result is represented.
        if viz_cfg["enabled"]:
            if (
                not snapshots
                or snapshots[-1]["result_set"] != nsets
            ):
                mapdl.set("LAST")
                ti = clean_temp(
                    np.asarray(
                        mapdl.post_processing.nodal_temperature(),
                        dtype=float,
                    )
                )
                snapshots.append(
                    {
                        "result_set": nsets,
                        "time_s": (
                            float(times[-1])
                            if times
                            else settings["duration"]
                        ),
                        "temperature_K": ti.copy(),
                    }
                )

        with hist_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "time_s",
                    "minimum_K",
                    "mean_K",
                    "maximum_K",
                ],
            )
            w.writeheader()
            w.writerows(history)

        with wall_hist_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "time_s",
                    "surface_order_index",
                    "x_m",
                    "y_m",
                    "applied_heat_flux_into_wall_W_m2",
                    "nearest_thermal_node_id",
                    "nearest_thermal_node_distance_m",
                    "wall_temperature_K",
                ],
            )
            w.writeheader()
            w.writerows(wall_history)

        visualization_outputs = {}
        visualization_error = None

        if viz_cfg["enabled"]:
            try:
                print(
                    "[thermal] Creating PDR visualization outputs..."
                )
                visualization_outputs = create_visualizations(
                    output_dir=output_dir,
                    boundary_xy=xy,
                    boundary_q=q,
                    xyz=xyz,
                    history=history,
                    snapshots=snapshots,
                    viz_cfg=viz_cfg,
                )
                for path in visualization_outputs.values():
                    print(f"[thermal] Visualization: {path}")
            except Exception as exc:
                visualization_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                print(
                    "[thermal] WARNING: visualization failed, "
                    "but the thermal solution is still valid."
                )
                print(
                    f"[thermal] Visualization error: "
                    f"{visualization_error}"
                )

        summary = {
            "status": "complete",
            "model_version": THERMAL_MODEL_VERSION,
            "loading_mode": "fixed_heat_flux",
            "screening_metrics": thermal_screening_metrics(
                xy, q, history, settings,
                ta_cfg.get("design_limits", {}).get("max_temperature_k"),
            ),
            "source_boundary_csv": str(bc_path),
            "geometry_source": "2-D polygon reconstructed from ordered CFD wall-face centroids",
            "wall_point_count": int(len(xy)),
            "thermal_mesh": {
                "element": "PLANE293",
                "form": "6-node triangular",
                "mesh_size_m": float(mesh_size),
                "nodes": int(n_nodes),
                "elements": int(n_elem),
            },
            "material": {
                "name": settings["material_name"],
                "conductivity_w_mk": settings["k"],
                "density_kg_m3": settings["rho"],
                "specific_heat_j_kgk": settings["cp"],
            },
            "analysis": {
                "type": "transient",
                "duration_s": settings["duration"],
                "initial_temperature_k": settings["initial_t"],
                "include_radiation": settings["include_rad"],
                "emissivity": settings["emissivity"],
                "radiation_temperature_k": settings["trad"],
            },
            "applied_heat_flux_W_m2": {
                "minimum": float(np.min(q)),
                "mean": float(np.mean(q)),
                "maximum": float(np.max(q)),
            },
            "final_temperature": final_stats,
            "outputs": {
                "thermal_nodes_final_csv": str(nodes_csv),
                "wall_temperature_final_csv": str(wall_csv),
                "temperature_history_csv": str(hist_csv),
                "wall_temperature_history_csv": str(wall_hist_csv),
                "visualization": visualization_outputs,
            },
            "limitations": [
                "The solid outline is reconstructed from CFD wall-face centroids, not exact thermal CAD.",
                "The Fluent cold-wall heat flux is applied as a fixed spatial load and is not updated as wall temperature changes.",
            ],
        }
        if visualization_error is not None:
            summary["visualization_error"] = visualization_error

        summary_json.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        rth_src = mapdl_run_dir / "thermal2d.rth"
        if rth_src.exists():
            rth_dst = output_dir / "thermal2d.rth"
            shutil.copy2(rth_src, rth_dst)
            summary["outputs"]["mapdl_result_file"] = str(rth_dst)
            summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("[thermal] Complete")
        print(
            f"[thermal] Final temperature range: {final_stats['minimum_K']:.3f} K "
            f"to {final_stats['maximum_K']:.3f} K"
        )
        print(f"[thermal] Results: {output_dir}")
        return summary

    finally:
        if mapdl is not None:
            try:
                mapdl.exit()
            except Exception:
                pass

def main():
    args = parse_args()

    return run_thermal_analysis(
        run_dir=args.run_dir,
        cfd_config=args.config,
        thermal_config=args.thermal_config,
    )


if __name__ == "__main__":
    try:
        main()
    except ThermalAnalysisError as exc:
        print("\n[thermal] CONFIGURATION / MODEL ERROR")
        print(exc)
        sys.exit(2)
