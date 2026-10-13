from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np
import yaml

from pipeline.trade_study import (
    AERO_FIELDS, THERMAL_METRIC_FIELDS, THERMAL_MODEL_VERSION,
    read_aero, aero_complete, create_trade_outputs,
)


class BatchError(RuntimeError):
    pass




def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise BatchError(f"Missing YAML file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise BatchError(f"{path} must contain a YAML mapping.")
    return data


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def safe_case_name(stem: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    if not value:
        raise BatchError(
            f"Could not make a case name from geometry stem {stem!r}."
        )
    return value


def validate_thermal_inputs(thermal_cfg: dict) -> None:
    mode = thermal_cfg.get("loading", {}).get("mode", "fixed_heat_flux")
    if mode != "fixed_heat_flux":
        raise BatchError(
            f"The supplied thermal runner cannot implement loading.mode={mode!r}. "
            "It applies fixed heat flux only. Provide the endpoint-convection runner, "
            "or explicitly select fixed_heat_flux for legacy screening."
        )
    material = thermal_cfg.get("material", {}) or {}
    transient = thermal_cfg.get("transient", {}) or {}

    required = {
        "material.conductivity_w_mk": material.get("conductivity_w_mk"),
        "material.density_kg_m3": material.get("density_kg_m3"),
        "material.specific_heat_j_kgk": material.get("specific_heat_j_kgk"),
        "transient.duration_s": transient.get("duration_s"),
    }

    missing = [
        name
        for name, value in required.items()
        if value is None
    ]
    if missing:
        raise BatchError(
            "run_thermal=true, but thermal_analysis.yaml still has required "
            "blank values:\\n  - "
            + "\\n  - ".join(missing)
            + "\\nFill them in, or set run_thermal: false in batch.yaml."
        )


def write_case_config(
    base_cfg: dict,
    geometry_file: Path,
    case_name: str,
    case_dir: Path,
    config_path: Path,
    force_fluent_gui,
) -> None:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))

    cfg.setdefault("project", {})
    cfg["project"]["name"] = case_name
    cfg["project"]["work_dir"] = str(case_dir.resolve())

    cfg.setdefault("geometry", {})
    cfg["geometry"]["input_file"] = str(geometry_file.resolve())

    if force_fluent_gui is not None:
        cfg.setdefault("runtime", {})
        cfg["runtime"]["fluent_gui"] = bool(force_fluent_gui)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def run_stage(
    root: Path,
    case_name: str,
    case_dir: Path,
    stage_name: str,
    args: list[str],
    hard_timeout_s: float | None = None,
    inactivity_timeout_s: float | None = None,
) -> None:
    """
    Run one pipeline stage while continuously teeing output to the case log.

    The reader thread prevents the parent process from blocking forever on a
    silent Fluent/MAPDL child.  A hard timeout limits total stage duration and
    an inactivity timeout catches the specific failure mode where a meshing RPC
    stops returning after SpaceClaim/geometry preparation.
    """
    log_dir = case_dir / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage_name}.log"

    command = [sys.executable, *args]

    print("")
    print("=" * 78)
    print(f"[batch:{case_name}] {stage_name}")
    print("[batch] " + " ".join(
        f'"{x}"' if " " in x else x
        for x in command
    ))
    if hard_timeout_s:
        print(f"[batch] Hard timeout: {hard_timeout_s:.0f} s")
    if inactivity_timeout_s:
        print(f"[batch] Inactivity timeout: {inactivity_timeout_s:.0f} s")
    print("=" * 78)

    with log_path.open(
        "w",
        encoding="utf-8",
        errors="replace",
    ) as log:
        process = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def _reader() -> None:
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        started = time.monotonic()
        last_output = started
        stream_done = False
        timed_out_reason = None

        while True:
            now = time.monotonic()
            if hard_timeout_s and now - started > hard_timeout_s:
                timed_out_reason = (
                    f"hard timeout after {hard_timeout_s:.0f} s"
                )
                break
            if (
                inactivity_timeout_s
                and now - last_output > inactivity_timeout_s
                and process.poll() is None
            ):
                timed_out_reason = (
                    "no console output for "
                    f"{inactivity_timeout_s:.0f} s"
                )
                break

            try:
                item = output_queue.get(timeout=0.5)
            except queue.Empty:
                item = ""

            if item is None:
                stream_done = True
            elif item:
                last_output = time.monotonic()
                print(item, end="")
                log.write(item)
                log.flush()

            if process.poll() is not None and stream_done:
                break

        if timed_out_reason is not None:
            message = (
                f"[batch] TIMEOUT: stage {stage_name!r} for {case_name} "
                f"stopped because {timed_out_reason}."
            )
            print(message)
            log.write(message + "\n")
            log.flush()
            _terminate_process_tree(process)
            raise BatchError(
                message
                + f" Log: {log_path}. The batch will continue according to "
                "stop_on_error."
            )

        return_code = process.wait()

    if return_code != 0:
        raise BatchError(
            f"Stage '{stage_name}' failed for {case_name} "
            f"with exit code {return_code}. Log: {log_path}"
        )


def cfd_complete(case_dir: Path) -> bool:
    return (
        (case_dir / "04_final.cas.h5").exists()
        and (case_dir / "04_final.dat.h5").exists()
    )


def _failed_during_meshing(case_dir: Path, stage_name: str = "01_cfd") -> bool:
    """Return True when a failed CFD stage never reached the solver launch."""
    log_path = case_dir / "batch_logs" / f"{stage_name}.log"
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    lowered = text.casefold()
    if any(word in lowered for word in (
        "attributeerror", "launchfluenterror", "launch process has timed out",
        "license", "filenotfounderror", "geometry service cannot be initialized",
    )):
        return False
    # A generic [mesh] message is not evidence that inflation was reached.
    return ("[solve] Launching" not in text and any(marker in lowered for marker in (
        "[mesh] adding boundary layers", "[mesh] generating boundary layers",
        "[mesh] boundary-layer", "[mesh] boundary layer",
    )))


def _disable_boundary_layer_in_case_config(config_path: Path) -> bool:
    """Disable 2-D inflation only in this case's generated batch config."""
    try:
        cfg = load_yaml(config_path)
        mesh = cfg.setdefault("mesh", {})
        bl = mesh.setdefault("boundary_layer", {})
        was_enabled = bool(bl.get("enabled", False))
        bl["enabled"] = False
        config_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False),
            encoding="utf-8",
        )
        return was_enabled
    except Exception:
        return False


def adiabatic_complete(case_dir: Path) -> bool:
    summary_path = case_dir / "adiabatic_reference" / "adiabatic_summary.json"
    csv_path = case_dir / "adiabatic_reference" / "adiabatic_wall_temperature.csv"
    if not summary_path.exists() or not csv_path.exists():
        return False
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("status") == "complete"


def thermal_bc_complete(
    case_dir: Path,
    base_cfg: dict,
) -> bool:
    bc_dir = (
        base_cfg.get("thermal_export", {})
        .get("output_directory", "thermal_bc")
    )
    return (
        case_dir
        / bc_dir
        / "body_wall_faces.csv"
    ).exists()


def thermal_complete(case_dir: Path) -> bool:
    summary_path = (
        case_dir
        / "thermal_analysis"
        / "thermal_summary.json"
    )
    if not summary_path.exists():
        return False
    try:
        data = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
    except Exception:
        return False
    return (
        data.get("status") == "complete"
        and data.get("model_version") == THERMAL_MODEL_VERSION
    )


def read_json(path: Path) -> dict:
    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return {}


def build_result_row(
    geometry_file: Path,
    case_name: str,
    case_dir: Path,
    status: str,
    elapsed_s: float,
    error: str = "",
) -> dict:
    row = {
        "geometry": geometry_file.name,
        "case": case_name,
        "status": status,
        "elapsed_s": round(float(elapsed_s), 3),
        "boundary_layer_used": "",
        "max_heat_flux_W_m2": "",
        "mean_heat_flux_W_m2": "",
        "adiabatic_max_wall_temperature_K": "",
        "final_max_temperature_K": "",
        "final_mean_temperature_K": "",
        "thermal_nodes": "",
        "thermal_elements": "",
        "error": error,
    }

    row.update({key: "" for key in (*AERO_FIELDS, *THERMAL_METRIC_FIELDS)})
    aero = read_aero(case_dir)
    row["aero_status"] = aero.get("status", "missing")
    row["aero_error"] = aero.get("error", "")
    row["coefficient_status"] = aero.get("coefficient_status", "")
    row["convergence_status"] = aero.get("convergence_status", "not_verified")
    if aero.get("status") == "complete":
        row.update({key: aero.get(key) for key in AERO_FIELDS})

    ad_summary = read_json(
        case_dir
        / "adiabatic_reference"
        / "adiabatic_summary.json"
    )
    if ad_summary:
        row["adiabatic_max_wall_temperature_K"] = (
            ad_summary.get("adiabatic_wall_temperature_K", {}) or {}
        ).get("maximum", "")

    case_cfg_path = case_dir / "_batch" / "config.yaml"
    try:
        case_cfg = load_yaml(case_cfg_path)
        row["boundary_layer_used"] = bool(
            case_cfg.get("mesh", {})
            .get("boundary_layer", {})
            .get("enabled", False)
        )
    except Exception:
        pass

    summary = read_json(
        case_dir
        / "thermal_analysis"
        / "thermal_summary.json"
    )
    row["thermal_loading_mode"] = summary.get("loading_mode", "unknown")
    row["thermal_comparison_signature"] = json.dumps({
        key: summary.get(key) for key in ("material", "analysis", "loading_mode")
    }, sort_keys=True)
    if summary:
        row.update({key: summary.get("screening_metrics", {}).get(key, "")
                    for key in THERMAL_METRIC_FIELDS})
        heat = (
            summary.get("applied_heat_flux_W_m2", {})
            or {}
        )
        temp = (
            summary.get("final_temperature", {})
            or {}
        )
        mesh = (
            summary.get("thermal_mesh", {})
            or {}
        )

        row["max_heat_flux_W_m2"] = heat.get(
            "maximum",
            "",
        )
        row["mean_heat_flux_W_m2"] = heat.get(
            "mean",
            "",
        )
        row["final_max_temperature_K"] = temp.get(
            "maximum_K",
            "",
        )
        row["final_mean_temperature_K"] = temp.get(
            "mean_K",
            "",
        )
        row["thermal_nodes"] = mesh.get("nodes", "")
        row["thermal_elements"] = mesh.get(
            "elements",
            "",
        )

    return row


def save_batch_summary(
    output_dir: Path,
    rows: list[dict],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "batch_summary.csv"
    json_path = output_dir / "batch_summary.json"

    fieldnames = [
        "geometry",
        "case",
        "status",
        "elapsed_s",
        "boundary_layer_used",
        "max_heat_flux_W_m2",
        "mean_heat_flux_W_m2",
        "adiabatic_max_wall_temperature_K",
        "final_max_temperature_K",
        "final_mean_temperature_K",
        "thermal_nodes",
        "thermal_elements",
        "error", "aero_status", "aero_error", "coefficient_status", "convergence_status",
        "thermal_loading_mode", "thermal_comparison_signature", *AERO_FIELDS, *THERMAL_METRIC_FIELDS,
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    return csv_path, json_path


def numeric(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def create_comparison_outputs(
    runs_dir: Path,
    rows: list[dict],
    output_dir: Path,
    base_cfg: dict,
) -> dict:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise BatchError(
            "Comparison plots require matplotlib. "
            "Install requirements.txt first."
        ) from exc

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    created = {}

    thermal_rows = [
        row
        for row in rows
        if numeric(
            row.get("final_max_temperature_K")
        ) is not None
    ]

    if thermal_rows:
        names = [
            row["case"]
            for row in thermal_rows
        ]

        valid_q = []
        valid_t = []
        scatter_rows = []

        for row in thermal_rows:
            qv = numeric(
                row.get("max_heat_flux_W_m2")
            )
            tv = numeric(
                row.get("final_max_temperature_K")
            )

            if qv is not None:
                valid_q.append(
                    (row["case"], qv)
                )
            if tv is not None:
                valid_t.append(
                    (row["case"], tv)
                )
            if qv is not None and tv is not None:
                scatter_rows.append(
                    (row["case"], qv, tv)
                )

        if valid_q:
            q_names = [x[0] for x in valid_q]
            q_values = [x[1] for x in valid_q]

            fig = plt.figure(
                figsize=(
                    max(8, 0.7 * len(q_names)),
                    5,
                )
            )
            ax = fig.add_subplot(111)
            ax.bar(q_names, q_values)
            ax.set_ylabel(
                "Maximum heat flux into wall [W/m²]"
            )
            ax.set_title(
                "Maximum Aerothermal Heat Flux by Nose Geometry"
            )
            ax.tick_params(
                axis="x",
                rotation=45,
            )
            fig.tight_layout()
            path = (
                output_dir
                / "max_heat_flux_by_geometry.png"
            )
            fig.savefig(path, dpi=220)
            plt.close(fig)
            created[
                "max_heat_flux_by_geometry"
            ] = str(path)

        if valid_t:
            t_names = [x[0] for x in valid_t]
            t_values = [x[1] for x in valid_t]

            fig = plt.figure(
                figsize=(
                    max(8, 0.7 * len(t_names)),
                    5,
                )
            )
            ax = fig.add_subplot(111)
            ax.bar(t_names, t_values)
            ax.set_ylabel(
                "Final maximum temperature [K]"
            )
            ax.set_title(
                "Final Maximum Temperature by Nose Geometry"
            )
            ax.tick_params(
                axis="x",
                rotation=45,
            )
            fig.tight_layout()
            path = (
                output_dir
                / "max_temperature_by_geometry.png"
            )
            fig.savefig(path, dpi=220)
            plt.close(fig)
            created[
                "max_temperature_by_geometry"
            ] = str(path)

        if scatter_rows:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111)

            for name, qv, tv in scatter_rows:
                ax.scatter(qv, tv)
                ax.annotate(
                    name,
                    (qv, tv),
                    xytext=(4, 4),
                    textcoords="offset points",
                )

            ax.set_xlabel(
                "Maximum heat flux into wall [W/m²]"
            )
            ax.set_ylabel(
                "Final maximum temperature [K]"
            )
            ax.set_title(
                "Thermal Response vs Aerothermal Heating"
            )
            ax.grid(True)
            fig.tight_layout()

            path = (
                output_dir
                / "max_temperature_vs_heat_flux.png"
            )
            fig.savefig(path, dpi=220)
            plt.close(fig)
            created[
                "max_temperature_vs_heat_flux"
            ] = str(path)

    # Shape overlay from CFD wall points.
    bc_dir_name = (
        base_cfg.get("thermal_export", {})
        .get("output_directory", "thermal_bc")
    )
    outlines = []

    for row in rows:
        case = row["case"]
        bc_path = (
            runs_dir
            / case
            / bc_dir_name
            / "body_wall_faces.csv"
        )
        if not bc_path.exists():
            continue

        try:
            with bc_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as f:
                records = list(csv.DictReader(f))

            records.sort(
                key=lambda r: int(
                    float(r["surface_order_index"])
                )
            )

            x = np.asarray(
                [float(r["x_m"]) for r in records],
                dtype=float,
            )
            y = np.asarray(
                [float(r["y_m"]) for r in records],
                dtype=float,
            )

            if len(x) < 3:
                continue

            # Preserve scale but translate every nose to a common origin.
            x = x - np.min(x)
            y = y - 0.5 * (
                np.min(y) + np.max(y)
            )
            outlines.append((case, x, y))
        except Exception:
            continue

    if outlines:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)

        for case, x, y in outlines:
            ax.plot(
                np.r_[x, x[0]],
                np.r_[y, y[0]],
                label=case,
            )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )
        ax.set_xlabel(
            "x from forward-most point [m]"
        )
        ax.set_ylabel(
            "y about geometry midline [m]"
        )
        ax.set_title("Nose Geometry Overlay")
        ax.grid(True)

        if len(outlines) <= 12:
            ax.legend()

        fig.tight_layout()
        path = (
            output_dir
            / "nose_geometry_overlay.png"
        )
        fig.savefig(path, dpi=220)
        plt.close(fig)
        created[
            "nose_geometry_overlay"
        ] = str(path)

    # PDR gallery of final temperature fields.
    image_paths = []

    for row in rows:
        case = row["case"]
        candidates = [
            (
                runs_dir
                / case
                / "thermal_analysis"
                / "visualization"
                / "final_temperature_field.png"
            ),
            (
                runs_dir
                / case
                / "thermal_plots"
                / "final_temperature_field.png"
            ),
        ]

        image_path = next(
            (
                p
                for p in candidates
                if p.exists()
            ),
            None,
        )
        if image_path is not None:
            image_paths.append(
                (case, image_path)
            )

    if image_paths:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            Image = None

        if Image is not None:
            thumb_w = 640
            thumb_h = 480
            label_h = 42
            cols = min(
                3,
                len(image_paths),
            )
            rows_n = math.ceil(
                len(image_paths) / cols
            )

            canvas = Image.new(
                "RGB",
                (
                    cols * thumb_w,
                    rows_n * (thumb_h + label_h),
                ),
                "white",
            )
            draw = ImageDraw.Draw(canvas)

            for i, (
                case,
                image_path,
            ) in enumerate(image_paths):
                with Image.open(image_path) as im:
                    tile = im.convert("RGB")
                    tile.thumbnail(
                        (thumb_w, thumb_h)
                    )

                    x0 = (
                        i % cols
                    ) * thumb_w
                    y0 = (
                        i // cols
                    ) * (
                        thumb_h + label_h
                    )

                    px = x0 + (
                        thumb_w - tile.width
                    ) // 2
                    py = y0 + (
                        thumb_h - tile.height
                    ) // 2

                    canvas.paste(
                        tile,
                        (px, py),
                    )

                    draw.text(
                        (
                            x0 + 10,
                            y0 + thumb_h + 10,
                        ),
                        case,
                        fill="black",
                    )

            path = (
                output_dir
                / "temperature_field_gallery.png"
            )
            canvas.save(path)
            created[
                "temperature_field_gallery"
            ] = str(path)

    created.update(create_trade_outputs(runs_dir, rows, output_dir))
    return created


def main(
    batch_config_path: str | Path = "batch.yaml",
) -> int:
    root = Path(__file__).resolve().parent
    batch_path = resolve(
        root,
        batch_config_path,
    )
    batch_cfg = load_yaml(batch_path)

    geometry_dir = resolve(
        root,
        batch_cfg.get(
            "geometry_directory",
            "./geometry_batch",
        ),
    )
    pattern = str(
        batch_cfg.get(
            "geometry_glob",
            "*.x_t",
        )
    )
    base_config_path = resolve(
        root,
        batch_cfg.get(
            "base_config",
            "./config.yaml",
        ),
    )
    thermal_config_path = resolve(
        root,
        batch_cfg.get(
            "thermal_config",
            "./thermal_analysis.yaml",
        ),
    )
    runs_dir = resolve(
        root,
        batch_cfg.get(
            "runs_directory",
            "./runs",
        ),
    )

    if not geometry_dir.exists():
        raise BatchError(
            "Geometry directory does not exist: "
            f"{geometry_dir}"
        )

    geometries = sorted(
        p.resolve()
        for p in geometry_dir.glob(pattern)
        if p.is_file()
    )

    if not geometries:
        raise BatchError(
            f"No geometry files matched {pattern!r} "
            f"in {geometry_dir}"
        )

    base_cfg = load_yaml(
        base_config_path
    )

    run_cfd = bool(
        batch_cfg.get(
            "run_cfd",
            True,
        )
    )
    ensure_bc = bool(
        batch_cfg.get(
            "ensure_thermal_bc",
            True,
        )
    )
    run_thermal = bool(
        batch_cfg.get(
            "run_thermal",
            True,
        )
    )
    run_extra_plots = bool(
        batch_cfg.get(
            "run_extra_static_plots",
            True,
        )
    )
    resume = bool(
        batch_cfg.get(
            "resume",
            True,
        )
    )
    stop_on_error = bool(
        batch_cfg.get(
            "stop_on_error",
            False,
        )
    )
    comparison_outputs = bool(
        batch_cfg.get(
            "comparison_outputs",
            True,
        )
    )
    force_gui = batch_cfg.get(
        "force_fluent_gui",
        None,
    )

    run_adiabatic = bool(
        batch_cfg.get("run_adiabatic_reference", True)
    )
    adiabatic_required = bool(
        batch_cfg.get("adiabatic_required", False)
    )
    retry_cfd_without_bl = bool(
        batch_cfg.get("retry_cfd_without_boundary_layer", True)
    )
    timeout_cfg = batch_cfg.get("timeouts", {}) or {}

    def _timeout(name: str, default):
        value = timeout_cfg.get(name, default)
        if value in (None, 0, 0.0, ""):
            return None
        value = float(value)
        return value if value > 0 else None

    cfd_hard_timeout = _timeout("cfd_hard_s", 7200)
    cfd_inactivity_timeout = _timeout("cfd_inactivity_s", 1200)
    export_hard_timeout = _timeout("thermal_export_hard_s", 1200)
    adiabatic_hard_timeout = _timeout("adiabatic_hard_s", 3600)
    adiabatic_inactivity_timeout = _timeout("adiabatic_inactivity_s", 1200)
    thermal_hard_timeout = _timeout("thermal_hard_s", 5400)
    plot_hard_timeout = _timeout("plot_hard_s", 600)

    if run_thermal:
        thermal_cfg = load_yaml(
            thermal_config_path
        )
        validate_thermal_inputs(
            thermal_cfg
        )

    names = [
        safe_case_name(p.stem)
        for p in geometries
    ]
    duplicates = sorted(
        {
            name
            for name in names
            if names.count(name) > 1
        }
    )
    if duplicates:
        raise BatchError(
            "Duplicate case names after filename "
            "sanitization: "
            + ", ".join(duplicates)
        )

    print("")
    print("[batch] Hypersonic nose trade study")
    print(
        f"[batch] Geometry directory: "
        f"{geometry_dir}"
    )
    print(
        f"[batch] Matched geometries: "
        f"{len(geometries)}"
    )
    print(
        f"[batch] Runs directory: "
        f"{runs_dir}"
    )
    print(
        "[batch] Chain: "
        f"CFD={run_cfd}, "
        f"thermal-BC={ensure_bc}, "
        f"adiabatic-reference={run_adiabatic}, "
        f"thermal={run_thermal}, "
        f"extra-plots={run_extra_plots}"
    )
    print(
        "[batch] Runs are sequential by design "
        "(one Fluent/MAPDL job at a time)."
    )

    rows = []
    comparison_dir = (
        runs_dir
        / "batch_comparison"
    )

    for index, geometry_file in enumerate(
        geometries,
        start=1,
    ):
        case_name = safe_case_name(
            geometry_file.stem
        )
        case_dir = (
            runs_dir
            / case_name
        )
        case_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        config_path = (
            case_dir
            / "_batch"
            / "config.yaml"
        )

        write_case_config(
            base_cfg=base_cfg,
            geometry_file=geometry_file,
            case_name=case_name,
            case_dir=case_dir,
            config_path=config_path,
            force_fluent_gui=force_gui,
        )

        print("")
        print("#" * 78)
        print(
            f"[batch] CASE "
            f"{index}/{len(geometries)}: "
            f"{case_name}"
        )
        print(
            f"[batch] Geometry: "
            f"{geometry_file}"
        )
        print("#" * 78)

        started = time.perf_counter()
        status = "complete"
        error = ""

        try:
            if run_cfd:
                if (
                    resume
                    and cfd_complete(case_dir)
                ):
                    print(
                        f"[batch:{case_name}] "
                        "CFD already complete; skipping."
                    )
                else:
                    try:
                        run_stage(
                            root,
                            case_name,
                            case_dir,
                            "01_cfd",
                            [
                                str(root / "run_all.py"),
                                "--config",
                                str(config_path),
                            ],
                            hard_timeout_s=cfd_hard_timeout,
                            inactivity_timeout_s=cfd_inactivity_timeout,
                        )
                    except Exception:
                        # Accuracy-first behavior: try the requested inflation
                        # mesh once.  If Fluent stalls/fails before the solver
                        # even launches, automatically retry this geometry
                        # without inflation instead of requiring manual repair.
                        can_retry = (
                            retry_cfd_without_bl
                            and _failed_during_meshing(case_dir, "01_cfd")
                            and _disable_boundary_layer_in_case_config(config_path)
                        )
                        if not can_retry:
                            raise
                        print(
                            f"[batch:{case_name}] Meshing failed/stalled with "
                            "the requested boundary layer. Retrying this case "
                            "once with boundary_layer.enabled=false."
                        )
                        run_stage(
                            root,
                            case_name,
                            case_dir,
                            "01b_cfd_retry_no_boundary_layer",
                            [
                                str(root / "run_all.py"),
                                "--config",
                                str(config_path),
                            ],
                            hard_timeout_s=cfd_hard_timeout,
                            inactivity_timeout_s=cfd_inactivity_timeout,
                        )
            elif not cfd_complete(case_dir):
                raise BatchError(
                    "run_cfd=false, but converged "
                    "case/data files do not exist "
                    f"for {case_name}."
                )

            if ensure_bc:
                if thermal_bc_complete(
                    case_dir,
                    base_cfg,
                ):
                    print(
                        f"[batch:{case_name}] "
                        "Thermal BC already exists."
                    )
                else:
                    # run_all.py handles geometry -> mesh -> Fluent solve
                    # and tries its thermal BC export, but it does NOT run
                    # MAPDL thermal analysis or create thermal plots.
                    #
                    # If the in-solver export did not create the CSV, retry
                    # from the already-saved case/data. This avoids rerunning
                    # the expensive CFD iterations.
                    run_stage(
                        root,
                        case_name,
                        case_dir,
                        "02_thermal_export",
                        [
                            str(
                                root
                                / "export_thermal_only.py"
                            ),
                            "--config",
                            str(config_path),
                        ],
                        hard_timeout_s=export_hard_timeout,
                        inactivity_timeout_s=None,
                    )

                if not thermal_bc_complete(
                    case_dir,
                    base_cfg,
                ):
                    raise BatchError(
                        "Thermal boundary-condition "
                        "export finished but "
                        "body_wall_faces.csv is "
                        "still missing."
                    )

            if base_cfg.get("aero", {}).get("enabled", True):
                if not (resume and aero_complete(case_dir, load_yaml(config_path))):
                    run_stage(
                        root, case_name, case_dir, "02b_aerodynamics",
                        [str(root / "export_aero_only.py"), "--config", str(config_path)],
                        hard_timeout_s=_timeout("aero_hard_s", 1200),
                        inactivity_timeout_s=_timeout("aero_inactivity_s", 600),
                    )
                if read_aero(case_dir).get("status") != "complete":
                    raise BatchError("Aerodynamic force export did not complete")

            if run_adiabatic:
                if resume and adiabatic_complete(case_dir):
                    print(
                        f"[batch:{case_name}] "
                        "Adiabatic reference already complete; skipping."
                    )
                else:
                    try:
                        run_stage(
                            root,
                            case_name,
                            case_dir,
                            "03_adiabatic_reference",
                            [
                                str(root / "run_adiabatic_reference.py"),
                                "--config",
                                str(config_path),
                            ],
                            hard_timeout_s=adiabatic_hard_timeout,
                            inactivity_timeout_s=adiabatic_inactivity_timeout,
                        )
                    except Exception as ad_exc:
                        if adiabatic_required:
                            raise
                        print(
                            f"[batch:{case_name}] WARNING: adiabatic reference "
                            f"failed; MAPDL will fall back to fixed cold-wall q''. "
                            f"{type(ad_exc).__name__}: {ad_exc}"
                        )

                if adiabatic_required and not adiabatic_complete(case_dir):
                    raise BatchError(
                        "Adiabatic reference was required but did not produce "
                        "adiabatic_wall_temperature.csv."
                    )

            if run_thermal:
                if (
                    resume
                    and thermal_complete(
                        case_dir
                    )
                ):
                    print(
                        f"[batch:{case_name}] "
                        "Thermal analysis already "
                        "complete; skipping MAPDL."
                    )
                else:
                    # Explicit stage after run_all.py:
                    # MAPDL + PNGs + heating GIF.
                    run_stage(
                        root,
                        case_name,
                        case_dir,
                        "04_thermal_analysis_and_visualization",
                        [
                            str(
                                root
                                / "run_thermal_analysis.py"
                            ),
                            "--config",
                            str(config_path),
                            "--thermal-config",
                            str(
                                thermal_config_path
                            ),
                        ],
                        hard_timeout_s=thermal_hard_timeout,
                        inactivity_timeout_s=None,
                    )

                if not thermal_complete(
                    case_dir
                ):
                    raise BatchError(
                        "Thermal analysis did not "
                        "produce a complete "
                        "thermal_summary.json."
                    )

                if run_extra_plots:
                    try:
                        run_stage(
                            root,
                            case_name,
                            case_dir,
                            "05_extra_static_plots",
                            [
                                str(
                                    root
                                    / "visualize_thermal_results.py"
                                ),
                                str(case_dir),
                            ],
                            hard_timeout_s=plot_hard_timeout,
                            inactivity_timeout_s=None,
                        )
                    except Exception as plot_exc:
                        print(
                            f"[batch:{case_name}] "
                            "WARNING: extra static "
                            "plotting failed: "
                            f"{plot_exc}"
                        )

        except Exception as exc:
            status = "failed"
            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )
            print(
                f"[batch:{case_name}] "
                f"FAILED: {error}"
            )

        elapsed = (
            time.perf_counter()
            - started
        )

        rows.append(
            build_result_row(
                geometry_file=geometry_file,
                case_name=case_name,
                case_dir=case_dir,
                status=status,
                elapsed_s=elapsed,
                error=error,
            )
        )

        save_batch_summary(
            comparison_dir,
            rows,
        )

        if (
            status == "failed"
            and stop_on_error
        ):
            print(
                "[batch] stop_on_error=true; "
                "stopping batch."
            )
            break

    csv_path, json_path = (
        save_batch_summary(
            comparison_dir,
            rows,
        )
    )

    comparison_created = {}
    if comparison_outputs:
        try:
            comparison_created = (
                create_comparison_outputs(
                    runs_dir=runs_dir,
                    rows=rows,
                    output_dir=comparison_dir,
                    base_cfg=base_cfg,
                )
            )
        except Exception as exc:
            print(
                "[batch] WARNING: comparison "
                "plotting failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    complete_count = sum(
        row["status"] == "complete"
        for row in rows
    )
    failed_count = sum(
        row["status"] == "failed"
        for row in rows
    )

    print("")
    print("=" * 78)
    print("[batch] FINISHED")
    print(
        f"[batch] Complete cases: "
        f"{complete_count}"
    )
    print(
        f"[batch] Failed cases:   "
        f"{failed_count}"
    )
    print(
        f"[batch] Summary CSV: "
        f"{csv_path}"
    )
    print(
        f"[batch] Summary JSON: "
        f"{json_path}"
    )

    for value in (
        comparison_created.values()
    ):
        print(
            "[batch] Comparison output: "
            f"{value}"
        )

    print("=" * 78)

    return (
        0
        if failed_count == 0
        else 2
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially run a directory of hypersonic "
            "nose geometries through CFD, thermal export, "
            "MAPDL thermal analysis, individual visualization, "
            "and cross-geometry comparison plotting."
        )
    )
    parser.add_argument(
        "--batch-config",
        default="batch.yaml",
        help="Batch YAML configuration file.",
    )
    args = parser.parse_args()

    try:
        sys.exit(
            main(args.batch_config)
        )
    except BatchError as exc:
        print(
            "\\n[batch] CONFIGURATION ERROR"
        )
        print(exc)
        sys.exit(2)
