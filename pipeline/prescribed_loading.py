"""Prescribed CFD equilibrium wall temperature, not convection or heat flux."""
from dataclasses import dataclass
from pathlib import Path
import copy

import numpy as np

from .thermal_loading import ThermalAnalysisError, _read_rows, validate_settings as material_settings


@dataclass
class PrescribedTemperatureLoad:
    xy: np.ndarray
    surface_temperature: np.ndarray
    mode: str = "prescribed_temperature"
    constrained_node_count: int | None = None


def validate_settings(cfd_cfg, cfg):
    if cfd_cfg.get("thermal", {}).get("mode") != "radiative_equilibrium":
        raise ThermalAnalysisError("Prescribed-temperature Mechanical loading requires radiative-equilibrium CFD.")
    if cfg.get("loading", {}).get("mode") != "prescribed_temperature":
        raise ThermalAnalysisError("Single-solve radiative-equilibrium CFD requires loading.mode=prescribed_temperature.")
    if cfg.get("include_radiation", False):
        raise ThermalAnalysisError("Disable include_radiation: the prescribed CFD surface temperature already includes radiation; Mechanical must not add RAD to that boundary.")
    compatible = copy.deepcopy(cfg)
    compatible["loading"] = {"mode": "fixed_heat_flux"}
    compatible["include_radiation"] = False
    settings = material_settings(cfd_cfg, compatible)
    settings.update(loading_mode="prescribed_temperature", boundary_application="step_at_t0_plus")
    return settings


def read_load(path):
    required = {"surface_order_index", "x_m", "y_m", "wall_temperature_K"}
    rows = _read_rows(Path(path), required)
    try:
        rows.sort(key=lambda row: int(row["surface_order_index"]))
        indices = [int(row["surface_order_index"]) for row in rows]
        data = np.asarray([[float(row[key]) for key in ("x_m", "y_m", "wall_temperature_K")] for row in rows])
    except (ValueError, TypeError) as exc:
        raise ThermalAnalysisError("Invalid boundary-node temperature CSV") from exc
    if indices != list(range(len(rows))) or len(rows) < 3 or not np.isfinite(data).all() or np.any(data[:, 2] <= 0):
        raise ThermalAnalysisError("Boundary nodes must be ordered, finite, and contain positive Kelvin temperatures.")
    xy = data[:, :2]
    if np.any(np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1) <= 1e-12):
        raise ThermalAnalysisError("Boundary contains duplicate consecutive nodes")
    area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
    if abs(area) <= 1e-16:
        raise ThermalAnalysisError("Boundary does not enclose a solid area")
    if area < 0:
        data = data[::-1].copy()
    return PrescribedTemperatureLoad(data[:, :2], data[:, 2])


def interpolate_boundary_temperature(load, node_xy, tolerance):
    """Interpolate Tw on the actual polygon segments, including midside nodes.

    Callers pass nodes selected with MAPDL NSLL, rather than selecting interior
    nodes merely because they lie close to the wall. Geometry is checked again
    here to avoid assigning temperature to an unrelated selected node.
    """
    node_xy = np.asarray(node_xy, dtype=float)
    if node_xy.ndim != 2 or node_xy.shape[1] != 2 or not np.isfinite(node_xy).all():
        raise ThermalAnalysisError("Invalid coordinates for selected Mechanical boundary nodes")
    starts = np.asarray(load.xy, dtype=float)
    vectors = np.roll(starts, -1, axis=0) - starts
    squared_lengths = np.sum(vectors * vectors, axis=1)
    if np.any(squared_lengths <= 1e-24):
        raise ThermalAnalysisError("Temperature boundary contains zero-length segments")
    values = np.empty(len(node_xy))
    for index, point in enumerate(node_xy):
        fraction = np.clip(np.sum((point - starts) * vectors, axis=1) / squared_lengths, 0, 1)
        projected = starts + fraction[:, None] * vectors
        distances = np.linalg.norm(projected - point, axis=1)
        segment = int(np.argmin(distances))
        if distances[segment] > tolerance:
            raise ThermalAnalysisError("A selected Mechanical boundary node does not lie on the reconstructed CFD wall")
        next_index = (segment + 1) % len(starts)
        values[index] = ((1 - fraction[segment]) * load.surface_temperature[segment]
                         + fraction[segment] * load.surface_temperature[next_index])
    return values
