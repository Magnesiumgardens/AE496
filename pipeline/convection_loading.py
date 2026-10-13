"""Temperature-driven Mechanical convection; no CFD heat-flux dependency."""
from dataclasses import dataclass
from pathlib import Path
import copy

import numpy as np

from .thermal_loading import ThermalAnalysisError, positive, validate_settings as legacy_settings, _read_rows


@dataclass
class ConvectionLoad:
    xy: np.ndarray
    gas_temperature: np.ndarray
    coefficient: float
    mode: str = "convection"

    @property
    def cold_flux(self):
        # Compatibility only for historical snapshot readers. Not a CFD load.
        return np.zeros(len(self.xy))

    def incoming_flux(self, wall_temperature):
        return self.coefficient * (self.gas_temperature - np.asarray(wall_temperature))


def validate_settings(cfd_cfg, cfg):
    if cfd_cfg.get("thermal") != {"mode":"adiabatic"}:
        raise ThermalAnalysisError("Mechanical convection requires adiabatic CFD temperatures.")
    loading = cfg.get("loading", {})
    if loading.get("mode") != "convection":
        raise ThermalAnalysisError("Set loading.mode=convection; CFD heat-flux loading is not used.")
    coefficient = positive("loading.heat_transfer_coefficient_w_m2k (h; no default)",
                           loading.get("heat_transfer_coefficient_w_m2k"))
    compatible = copy.deepcopy(cfg)
    compatible["loading"] = {"mode":"fixed_heat_flux"}
    settings = legacy_settings(cfd_cfg, compatible)
    settings.update(loading_mode="convection", h=coefficient)
    return settings


def read_load(path, coefficient):
    rows = _read_rows(Path(path), {"surface_order_index", "x_m", "y_m", "recovery_temperature_K"})
    try:
        rows.sort(key=lambda row:int(row["surface_order_index"]))
        indices = [int(row["surface_order_index"]) for row in rows]
        values = np.asarray([[float(row[key]) for key in ("x_m","y_m","recovery_temperature_K")] for row in rows])
    except (ValueError, TypeError) as exc:
        raise ThermalAnalysisError("Invalid boundary-node temperature CSV") from exc
    if indices != list(range(len(rows))) or len(rows) < 3 or not np.isfinite(values).all() or np.any(values[:,2] <= 0):
        raise ThermalAnalysisError("Boundary nodes must be ordered, finite, and contain positive Kelvin temperatures.")
    xy = values[:,:2]
    if np.any(np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1) <= 1e-12):
        raise ThermalAnalysisError("Boundary contains duplicate consecutive nodes")
    area = np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))
    if abs(area) <= 1e-16:raise ThermalAnalysisError("Boundary does not enclose a solid area")
    if area < 0:values = values[::-1].copy()
    return ConvectionLoad(values[:,:2], values[:,2], coefficient)
