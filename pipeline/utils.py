from pathlib import Path
import json
import os
import tempfile


def work_paths(cfg):
    work = Path(cfg["project"]["work_dir"]).resolve()
    work.mkdir(parents=True, exist_ok=True)
    return {
        "work": work, "labeled_geometry": work / "01_labeled_geometry.pmdb",
        "mesh": work / "02_mesh.msh.h5", "setup_case": work / "03_setup.cas.h5",
        "final_case": work / "04_final.cas.h5", "final_data": work / "04_final.dat.h5",
        "summary": work / "summary.json", "heat_flux": work / "body_wall_heat_flux.csv",
    }


def safe_allowed_values(setting):
    try:
        return list(setting.allowed_values())
    except (AttributeError, RuntimeError):
        return []


def choose_case_insensitive(values, requested):
    return next((value for value in values if str(value).casefold() == requested.casefold()), None)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
