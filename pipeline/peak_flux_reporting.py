"""Compare peaks in saved surface-flux CSVs without running an Ansys engine."""
from __future__ import annotations

import hashlib
import fnmatch
import json
from pathlib import Path

import numpy as np

from .reporting import _plotting, load_yaml, numeric, read_csv, write_csv
from .surface_reporting import surface_load_bar_outputs
from .utils import atomic_json, file_sha256

FLUX_FIELDS = (
    "convective_heat_flux_into_solid_W_m2",
    "radiative_heat_flux_out_W_m2",
    "net_heat_flux_into_solid_W_m2",
)


def _json_if_present(path):
    if not path.is_file():return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):raise ValueError(f"Invalid saved receipt: {path.name}")
    return value


def _saved_flux(work):
    folder = work / "thermal_analysis"
    source = folder / "surface_heat_flux_history.csv"
    if not source.is_file():source = folder / "wall_temperature_history.csv"
    if not source.is_file():raise ValueError("No saved surface heat-flux history CSV")
    summary = _json_if_present(folder / "thermal_summary.json")
    energy = _json_if_present(folder / "energy_balance.json")
    if (summary.get("status") in {"failed", "running"}
            or summary.get("energy_balance", {}).get("status") == "failed" or energy.get("status") == "failed"):
        raise ValueError("Saved thermal/energy check failed or analysis is still running")
    digest = file_sha256(source)
    declared = summary.get("result_sha256", {}).get(source.name)
    if declared and digest != declared:raise ValueError("Saved heat-flux CSV hash does not match its receipt")
    energy_digest = summary.get("result_sha256", {}).get("energy_balance.json")
    if energy and energy_digest and file_sha256(folder / "energy_balance.json") != energy_digest:
        raise ValueError("Saved energy-check hash does not match its receipt")
    records = read_csv(source)
    if not records or not {"time_s", "surface_order_index", *FLUX_FIELDS} <= set(records[0]):
        raise ValueError("Saved CSV lacks heat-flux columns; temperature alone cannot supply heat flux")
    selected, identities = [], set()
    for original in records:
        record = dict(original)
        # Outgoing segment flags are not node activity flags: a shared rear
        # corner still belongs to an exposed forward edge and remains included.
        if "thermal_active" in record:
            active = numeric(record["thermal_active"])
            if active not in (0., 1.):raise ValueError("Invalid thermal-active flag")
            if active == 0.:continue
        for field in ("time_s", *FLUX_FIELDS):
            value = numeric(record.get(field))
            if value is None:raise ValueError(f"Missing or nonfinite {field}")
            record[field] = value
        index = numeric(record["surface_order_index"])
        if record["time_s"] < 0 or index is None or index < 0 or not float(index).is_integer():
            raise ValueError("Invalid physical time or surface sample index")
        record["surface_order_index"] = int(index)
        identity = (record["time_s"], record["surface_order_index"])
        if identity in identities:raise ValueError("Duplicate time/surface sample in heat-flux CSV")
        identities.add(identity)
        for field in ("x_m", "y_m"):
            if field in record and numeric(record[field]) is None:raise ValueError(f"Nonfinite {field}")
        if not np.isclose(record[FLUX_FIELDS[2]], record[FLUX_FIELDS[0]] - record[FLUX_FIELDS[1]], rtol=1e-8, atol=1e-6):
            raise ValueError("Net flux does not equal convection minus radiation")
        selected.append(record)
    if not selected:raise ValueError("No active surface samples in saved heat-flux CSV")
    selected.sort(key=lambda row: (row["time_s"], row["surface_order_index"]))
    validation = "saved_csv_hash_matched" if declared else "saved_receipt_only" if summary else "saved_csv_only"
    conditions = dict(
        conditions=summary.get("surface_load_conditions", summary.get("temperature_conditions")),
        settings=summary.get("settings"), loading_mode=summary.get("loading_mode"),
        model_version=summary.get("model_version"),
        rear_boundary=summary.get("thermal_surface_scope", {}).get("rear_boundary"),
        first_time_s=selected[0]["time_s"], final_time_s=selected[-1]["time_s"],
        receipt_available=bool(summary), validation_status=validation,
    )
    key = json.dumps(conditions, sort_keys=True, allow_nan=False)
    data = dict(flux=selected, source_csv=str(source), source_sha256=digest,
                validation_status=validation, conditions=conditions, key=key,
                nose_direction=summary.get("settings", {}).get("thermal_surface_selection", {}).get("nose_direction", [1., 0.]))
    config_path = work / "_batch/config.yaml"
    if config_path.is_file():
        try:
            data["nose_direction"] = load_yaml(config_path).get("reporting", {}).get("nose_direction", data["nose_direction"])
        except (OSError, ValueError, TypeError, AttributeError):
            data["nose_direction"] = None  # Spatial plotting will explain this; peak values remain usable.
    return data


def _geometry_cases(runs, geometry_dir, geometry_glob, cases):
    """Use the batch's CAD-to-safe-case mapping without opening or meshing CAD."""
    from .study import safe_case_name
    patterns = [pattern.casefold() for pattern in cases] if cases else []
    def selected(*names):
        return not patterns or any(fnmatch.fnmatchcase(name.casefold(), pattern) for name in names for pattern in patterns)
    if geometry_dir is None and (runs.parent / "geometry_batch").is_dir():
        geometry_dir = runs.parent / "geometry_batch"
    if geometry_dir is None:
        return [work for work in sorted(runs.iterdir()) if work.is_dir()
                and (work / "thermal_analysis").is_dir() and selected(work.name)], [], None
    geometry = Path(geometry_dir).resolve()
    if not geometry.is_dir():raise ValueError(f"Geometry directory does not exist: {geometry}")
    inputs = sorted(path for path in geometry.glob(geometry_glob) if path.is_file())
    if not inputs:raise ValueError(f"No geometry inputs matching {geometry_glob!r} under {geometry}")
    run_names = {}
    for work in runs.iterdir():
        if work.is_dir():run_names.setdefault(work.name.casefold(), []).append(work)
    candidates, inventory, identities = [], [], set()
    for path in inputs:
        name = safe_case_name(path.stem)
        identity = name.casefold()
        if identity in identities:raise ValueError(f"CAD filenames produce colliding case names: {name}")
        identities.add(identity)
        chosen = selected(name, path.stem, path.name)
        matches = run_names.get(identity, [])
        if chosen and len(matches) > 1:raise ValueError(f"Multiple saved run folders match CAD case {name}")
        work = matches[0] if len(matches) == 1 else runs / name
        status = "not_selected" if not chosen else "selected" if (work / "thermal_analysis").is_dir() else "missing_saved_results"
        inventory.append(dict(geometry_file=str(path), case=name, run_case=work.name if matches else None, status=status))
        if chosen:candidates.append(work)
    if patterns and not candidates:
        available = ", ".join(item["case"] for item in inventory)
        raise ValueError(f"No geometry cases match {cases!r}. Available CAD cases: {available}")
    return candidates, inventory, str(geometry)


def _check_geometry(work, geometry_file):
    """Check available CAD provenance without requiring absent archive inputs."""
    metadata_path = work / "thermal_bc/surface_load_summary.json"
    metadata = _json_if_present(metadata_path)
    receipt = _json_if_present(work / "thermal_analysis/thermal_summary.json")
    recorded_package = receipt.get("surface_package_sha256", receipt.get("source_surface_load_summary_sha256"))
    if metadata and recorded_package and file_sha256(metadata_path) != recorded_package:
        raise ValueError("Saved thermal results do not match the available calibration package for this geometry")
    recorded_geometry = metadata.get("source_geometry_sha256")
    if recorded_geometry:
        if file_sha256(geometry_file) != recorded_geometry:
            raise ValueError("CAD geometry changed since the saved heat-flux calibration; existing results belong to a different geometry")
        return "saved_calibration_CAD_hash_matched"
    return "filename_matched_CAD_provenance_unavailable"


def report_peak_heat_flux(runs_dir, output_dir=None, *, geometry_dir=None, geometry_glob="*.x_t", cases=None):
    """Report available saved samples, preserving known failure/provenance flags.

    This is an archival CSV report, not a fresh native-solution validation. It
    requires neither native binaries nor calibration inputs. It never converts
    temperatures or cold-wall calibration anchors into transient flux histories.
    """
    runs = Path(runs_dir).resolve()
    if not runs.is_dir():raise ValueError(f"Runs directory does not exist: {runs}")
    candidates, inventory, selected_geometry = _geometry_cases(runs, geometry_dir, geometry_glob, cases)
    output = Path(output_dir).resolve() if output_dir else runs / "comparison/peak_heat_flux"
    output.mkdir(parents=True, exist_ok=True)
    datasets, grouped, excluded = {}, {}, []
    geometry_files = {item["run_case"] or item["case"]: item["geometry_file"]
                      for item in inventory if item["status"] != "not_selected"}
    for work in candidates:
        try:
            data = _saved_flux(work)
            if work.name in geometry_files:
                data["geometry_file"] = geometry_files[work.name]
                data["geometry_check"] = _check_geometry(work, geometry_files[work.name])
            datasets[work.name] = data
            grouped.setdefault(data["key"], []).append(work.name)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            excluded.append(dict(case=work.name, reason=str(exc)))
            print(f"[peak-flux] {work.name}: excluded: {exc}")
    # Remove only artifacts owned by this command, never source CSVs/results.
    artifacts = ("peak_heat_flux_comparison", "peak_heat_flux_spatial_envelope",
                 "initial_heat_flux_spatial_comparison", "final_heat_flux_spatial_comparison")
    for stem in (name + suffix for name in artifacts for suffix in (".png", ".csv")):
        (output / stem).unlink(missing_ok=True)
        for path in output.glob("conditions_*/" + stem):path.unlink()
    plt = _plotting()
    colors = {name: plt.get_cmap("tab10")(i % 10) for i, name in enumerate(datasets)}
    groups = []
    for key, names in grouped.items():
        target = output if len(grouped) == 1 else output / ("conditions_" + hashlib.sha256(key.encode()).hexdigest()[:12])
        target.mkdir(parents=True, exist_ok=True)
        files = surface_load_bar_outputs({name: datasets[name] for name in names}, target, colors)
        table = Path(files["peak_heat_flux_comparison_csv"])
        rows = read_csv(table)
        for row in rows:
            data = datasets[row["design"]]
            row.update(source_csv=data["source_csv"], source_sha256=data["source_sha256"],
                       validation_status=data["validation_status"],
                       geometry_file=data.get("geometry_file", ""), geometry_check=data.get("geometry_check", "not_selected_from_CAD"),
                       first_time_s=data["flux"][0]["time_s"],
                       sample_count=len(data["flux"]), includes_time_zero=data["flux"][0]["time_s"] == 0.)
        write_csv(table, rows)
        from .spatial_flux_reporting import spatial_flux_outputs
        spatial = spatial_flux_outputs({name: datasets[name] for name in names}, target, colors)
        files.update(spatial["files"])
        for artifact, path in spatial["files"].items():
            if artifact.endswith("_png"):print(f"[peak-flux] Spatial comparison: {path}")
        groups.append(dict(output_dir=str(target), cases=names, conditions=json.loads(key), files=files,
                           spatial_excluded=spatial["excluded"]))
        for issue in spatial["excluded"]:print(f"[peak-flux] {issue['case']}: spatial profile unavailable: {issue['reason']}")
        print(f"[peak-flux] {len(names)} nose(s): {files['peak_heat_flux_comparison_png']}")
    manifest = output / "peak_heat_flux_report.json"
    report = dict(status="complete" if datasets and not excluded else "partial" if datasets else "unavailable",
        case_count=len(datasets), groups=groups, excluded=excluded,
        geometry_directory=selected_geometry, geometry_inventory=inventory, requested_cases=cases or [],
        peak_definition="Maximum signed local flux over all available saved surface samples and physical times; components peak independently",
        units="W/m2; not integrated power W/m or accumulated heat load J/m2",
        verification="Saved CSVs and available receipt hashes only; no fresh CFD, native-file or complete Mechanical validation",
        unknown_conditions="CSV-only archives have unverified run conditions and are separated from receipt-backed data",
        scope="Preserve included boundary vertices, including shared corners whose outgoing segment flag is zero",
        spatial_interpretation="Initial/final snapshots and maximum-at-each-location envelopes on each nose's own back-to-nose-to-back coordinates; no interpolation between geometries")
    atomic_json(manifest, report)
    print(f"[peak-flux] Report: {manifest}")
    return dict(report, manifest=str(manifest))
