"""Regenerate solid reports only from current temperature-driven analyses."""
from pathlib import Path

from .reporting import read_json, load_yaml, build_result_row, create_comparison_outputs
from .thermal_loading import ThermalAnalysisError
from .utils import file_sha256


def mechanical_complete(work, cfg):
    try:
        from .gas_reporting import temperature_complete
        from .solution_quality import saved_quality
        from .temperature_export import temperature_paths
        work = Path(work)
        summary = read_json(work / "thermal_analysis/thermal_summary.json")
        endpoint = cfg.get("workflow", {}).get("strategy") == "two_endpoint"
        prescribed = cfg.get("thermal", {}).get("mode") == "radiative_equilibrium"
        model = "5.0-prescribed-temperature" if prescribed else ("4.0-endpoint-convection" if endpoint else "3.0-temperature-convection")
        if summary.get("model_version") != model or summary.get("status") != "complete":return False
        if prescribed and summary.get("loading_mode") != "prescribed_temperature":return False
        if file_sha256(summary["thermal_config_path"]) != summary["thermal_config_sha256"]:return False
        if endpoint:
            from .endpoint_loading import verify_convection_package, convection_paths
            boundary, metadata = verify_convection_package(cfg)
            if summary.get("source_sha256") != metadata["source_sha256"]:return False
            if summary.get("convection_package_sha256") != file_sha256(convection_paths(cfg)[1]):return False
            if summary.get("convection_conditions") != metadata["conditions"]:return False
        else:
            if not temperature_complete(cfg) or saved_quality(cfg)["status"] != "residual_screen_passed":return False
            if summary["source_sha256"] != {"final_case":file_sha256(work/"04_final.cas.h5"),
                                             "final_data":file_sha256(work/"04_final.dat.h5")}:return False
            boundary = temperature_paths(cfg)[0].parent/"wall_temperature_nodes.csv"
            if prescribed:
                temperature_summary = temperature_paths(cfg)[1]
                if summary.get("source_temperature_summary_sha256") != file_sha256(temperature_summary):return False
                metadata = read_json(temperature_summary).get("wall_nodes", {})
                if metadata.get("status") != "complete" or metadata.get("csv_sha256") != file_sha256(boundary):return False
        if file_sha256(boundary) != summary["boundary_csv_sha256"]:return False
        return all(file_sha256(work/"thermal_analysis"/name) == digest for name,digest in summary["result_sha256"].items())
    except (OSError, ValueError, KeyError, TypeError, ThermalAnalysisError):return False


def create_mechanical_reports(runs_dir, output_dir):
    runs = Path(runs_dir)
    rows, groups = [], {}
    for work in sorted(runs.iterdir()) if runs.is_dir() else []:
        path = work / "_batch/config.yaml"
        if not path.is_file():continue
        cfg = load_yaml(path)
        if mechanical_complete(work,cfg):
            row = build_result_row(cfg["geometry"]["input_file"],work.name,work,"complete",0)
            rows.append(row)
            import json
            from .temperature_export import temperature_conditions
            summary = read_json(work / "thermal_analysis/thermal_summary.json")
            key = json.dumps({"cfd":temperature_conditions(cfg),"solid":summary["settings"]},sort_keys=True)
            groups.setdefault(key,[]).append(row)
    # Only this module owns the new comparison/solid folder. Clear previous
    # generated figures so stale or incompatible cases cannot remain displayed.
    import shutil
    if Path(output_dir).is_dir():shutil.rmtree(output_dir)
    if not rows:return [], {}
    if len(groups) == 1:
        return rows, create_comparison_outputs(runs,rows,output_dir,case_names={row["case"] for row in rows})
    import hashlib
    outputs = {}
    # Different flow/material/h/exposure settings get separate comparisons.
    for key, group in groups.items():
        name = "conditions_" + hashlib.sha256(key.encode()).hexdigest()[:12]
        outputs[name] = create_comparison_outputs(runs,group,Path(output_dir)/name,
                                                  case_names={row["case"] for row in group})
    return rows, outputs
