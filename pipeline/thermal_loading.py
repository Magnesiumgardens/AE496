"""Validate and map exported CFD loads without requiring an Ansys license."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ThermalAnalysisError(RuntimeError):
    pass


def positive(name, value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ThermalAnalysisError(f"{name} must be a positive number.") from exc
    if not np.isfinite(number) or number <= 0:
        raise ThermalAnalysisError(f"{name} must be a finite positive number.")
    return number


def validate_settings(cfd_cfg: dict, ta_cfg: dict) -> dict:
    loading = ta_cfg.get("loading", {}) or {}
    mode = loading.get("mode", "fixed_heat_flux")
    if mode != "fixed_heat_flux":
        raise ThermalAnalysisError("This pipeline applies direct CFD heat flux. Set loading.mode=fixed_heat_flux; legacy endpoint/convection modes are unsupported.")
    material = ta_cfg.get("material", {})
    transient = ta_cfg.get("transient", {})
    thermal = cfd_cfg.get("thermal", {})
    def default(value, fallback):
        return fallback if value is None else value
    duration = positive("transient.duration_s", transient.get("duration_s"))
    dt0 = positive("initial_time_step_s", default(transient.get("initial_time_step_s"), duration / 100))
    dtmin = positive("min_time_step_s", default(transient.get("min_time_step_s"), dt0 / 10))
    dtmax = positive("max_time_step_s", default(transient.get("max_time_step_s"), min(duration, dt0 * 10)))
    if not dtmin <= dt0 <= dtmax <= duration:
        raise ThermalAnalysisError("Require min_time_step_s <= initial_time_step_s <= max_time_step_s <= duration_s.")
    emissivity = float(default(ta_cfg.get("emissivity"), thermal.get("emissivity", 0.9)))
    if not np.isfinite(emissivity) or not 0 <= emissivity <= 1:
        raise ThermalAnalysisError("emissivity must be between 0 and 1.")
    return {
        "loading_mode": mode,
        "material_name": str(material.get("name", "user_material")),
        "k": positive("material.conductivity_w_mk", material.get("conductivity_w_mk")),
        "rho": positive("material.density_kg_m3", material.get("density_kg_m3")),
        "cp": positive("material.specific_heat_j_kgk", material.get("specific_heat_j_kgk")),
        "duration": duration, "dt0": dt0, "dtmin": dtmin, "dtmax": dtmax,
        "initial_t": positive("initial_temperature_k", default(ta_cfg.get("initial_temperature_k"), thermal.get("wall_temperature_k", 300))),
        "include_rad": bool(ta_cfg.get("include_radiation", True)),
        "emissivity": emissivity,
        "trad": positive("radiation_temperature_k", default(ta_cfg.get("radiation_temperature_k"), thermal.get("external_radiation_temperature_k", 3))),
    }


def _read_rows(path: Path, required: set[str]) -> list[dict]:
    if not path.is_file():
        raise ThermalAnalysisError(f"Thermal boundary CSV does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ThermalAnalysisError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ThermalAnalysisError(f"Boundary CSV is empty: {path}")
    return rows


def read_boundary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    fields = {"surface_order_index", "x_m", "y_m", "heat_flux_into_wall_W_m2"}
    try:
        rows = sorted(_read_rows(path, fields), key=lambda row: int(row["surface_order_index"]))
        indices = [int(row["surface_order_index"]) for row in rows]
        data = np.asarray([[float(row[name]) for name in ("x_m", "y_m", "heat_flux_into_wall_W_m2")] for row in rows])
    except (ValueError, TypeError) as exc:
        raise ThermalAnalysisError(f"Invalid numeric boundary data in {path}.") from exc
    if len(indices) != len(set(indices)) or not np.isfinite(data).all():
        raise ThermalAnalysisError("Boundary order indices must be unique and coordinates/flux finite.")
    tolerance = max(1e-12, float(np.ptp(data[:, :2], axis=0).max()) * 1e-10)
    keep = [0]
    for index in range(1, len(data)):
        if np.linalg.norm(data[index, :2] - data[keep[-1], :2]) > tolerance:
            keep.append(index)
    if len(keep) > 1 and np.linalg.norm(data[keep[-1], :2] - data[0, :2]) <= tolerance:
        keep.pop()
    data = data[keep]
    xy = data[:, :2]
    if len(xy) < 3:
        raise ThermalAnalysisError("Need at least three unique ordered wall points.")
    area2 = float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1)))
    if abs(area2) <= tolerance * tolerance:
        raise ThermalAnalysisError("Wall coordinates do not enclose a nonzero area.")
    if area2 < 0:
        data = data[::-1].copy()
    return data[:, :2], data[:, 2]


@dataclass
class ThermalLoad:
    xy: np.ndarray
    cold_flux: np.ndarray
    mode: str = "fixed_heat_flux"

    def incoming_flux(self, wall_temperature):
        return self.cold_flux.copy()


def prepare_load(boundary_path: Path, cfd_cfg: dict, ta_cfg: dict, thermal_config_dir: Path) -> ThermalLoad:
    if (ta_cfg.get("loading", {}) or {}).get("mode", "fixed_heat_flux") != "fixed_heat_flux":
        raise ThermalAnalysisError("Direct heat-flux loading is required; no endpoint workflow is generated.")
    xy, q = read_boundary(boundary_path)
    return ThermalLoad(xy, q)


def integrate_boundary(xy, values):
    lengths = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    return float(np.sum(0.5 * (values + np.roll(values, -1)) * lengths))


def auto_mesh_size(xy, configured):
    if configured is not None:
        return positive("mesh_size_m", configured)
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    span = float(np.ptp(xy, axis=0).max())
    return min(max(2 * float(np.median(ds[ds > 0])), span / 500), span / 40)
