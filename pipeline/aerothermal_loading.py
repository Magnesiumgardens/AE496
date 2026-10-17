"""Finite, wall-temperature-dependent heating calibrated from two wall states.

The fitted incoming law is ``q_conv = source - H * T_wall``.  MAPDL's native
CONV load evaluates that feedback implicitly, and RAD evaluates surface cooling
at the current solved temperature.  No temperature constraint is used.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .thermal_loading import ThermalAnalysisError, _read_rows, validate_settings as material_settings

STEFAN_BOLTZMANN = 5.670374419e-8


@dataclass
class AerothermalLoad:
    xy: np.ndarray
    coefficient: np.ndarray
    aerodynamic_source: np.ndarray
    equilibrium_temperature: np.ndarray
    cold_temperature: np.ndarray
    cold_flux: np.ndarray
    equilibrium_flux: np.ndarray
    mode: str = "aerodynamic_surface_load"
    boundary_quadrature: list = field(default_factory=list)

    @property
    def gas_temperature(self):
        """Effective affine-law intercept temperature, not an exported gas field."""
        return self.aerodynamic_source / self.coefficient

    def incoming_flux(self, wall_temperature):
        return self.aerodynamic_source - self.coefficient * np.asarray(wall_temperature)


def validate_settings(cfd_cfg, cfg):
    if cfd_cfg.get("thermal", {}).get("mode") != "radiative_equilibrium":
        raise ThermalAnalysisError("Aerodynamic surface loading requires radiative-equilibrium primary CFD.")
    if cfg.get("loading", {}).get("mode") != "aerodynamic_surface_load":
        raise ThermalAnalysisError("Set loading.mode=aerodynamic_surface_load for calibrated transient surface heating.")
    if cfg.get("include_radiation") is not True:
        raise ThermalAnalysisError("Aerodynamic surface loading requires include_radiation: true; the calibrated incoming load is gas convection only.")
    compatible = copy.deepcopy(cfg)
    compatible["loading"] = {"mode": "fixed_heat_flux"}
    settings = material_settings(cfd_cfg, compatible)
    thermal = cfd_cfg["thermal"]
    for name, actual, expected in (
        ("emissivity", settings["emissivity"], thermal.get("emissivity")),
        ("radiation_temperature_k", settings["trad"], thermal.get("external_radiation_temperature_k")),
    ):
        try:
            expected = float(expected)
        except (TypeError, ValueError) as exc:
            raise ThermalAnalysisError(f"CFD must define a finite {name} for the equilibrium anchor.") from exc
        if not np.isfinite(expected) or not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
            raise ThermalAnalysisError(f"Mechanical {name} must match the CFD radiation-equilibrium calibration ({expected}).")
    settings.update(loading_mode="aerodynamic_surface_load", boundary_application="implicit_CONV_plus_RAD",
                    temperature_history_sampling="solved_solid_temperature", calibration_model="affine_two_wall_states")
    return settings


def read_load(path):
    names = ("x_m", "y_m", "aerodynamic_conductance_W_m2K", "aerodynamic_source_W_m2",
             "equilibrium_temperature_K", "cold_wall_temperature_K", "cold_heat_flux_into_wall_W_m2",
             "equilibrium_convective_heat_flux_W_m2", "driving_temperature_K")
    rows = _read_rows(Path(path), {"surface_order_index", *names})
    try:
        rows.sort(key=lambda row: int(row["surface_order_index"]))
        order = [int(row["surface_order_index"]) for row in rows]
        data = np.asarray([[float(row[name]) for name in names] for row in rows])
    except (TypeError, ValueError) as exc:
        raise ThermalAnalysisError("Invalid calibrated aerodynamic surface-load CSV.") from exc
    if order != list(range(len(rows))) or len(rows) < 3 or not np.isfinite(data).all():
        raise ThermalAnalysisError("Calibrated surface nodes must have consecutive indices and finite values.")
    if (data[:, 2] <= 0).any() or (data[:, 4:6] <= 0).any() or (data[:, 8] <= 0).any():
        raise ThermalAnalysisError("Calibrated H and absolute temperatures must be positive; no coefficient clipping is performed.")
    if not np.allclose(data[:, 3] / data[:, 2], data[:, 8], rtol=1e-9, atol=1e-7):
        raise ThermalAnalysisError("Calibrated source/H does not match the supplied effective driving temperature.")
    xy = data[:, :2]
    if (np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1) <= 1e-12).any():
        raise ThermalAnalysisError("Calibrated boundary contains duplicate consecutive vertices.")
    area2 = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
    if abs(area2) <= 1e-16:
        raise ThermalAnalysisError("Calibrated boundary does not enclose a solid area.")
    if area2 < 0:
        data = data[::-1].copy()
    return AerothermalLoad(data[:, :2], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6], data[:, 7])


def prepare_boundary_quadrature(mapdl, lines, load, nnum, xyz, mesh_size):
    """Identify every quadratic FE boundary edge for postprocessed heat energy.

    PLANE293 six-node triangles give one midside node between adjacent corner
    nodes on each straight boundary line. Five-point Gaussian quadrature
    integrates the radiation polynomial of the quadratic temperature field.
    This is an independent boundary integral, not native heat-reaction output.
    """
    mapdl.lsel("NONE")
    for line in lines:
        mapdl.lsel("A", "LINE", "", int(line))
    mapdl.nsll("S", 1)
    selected = np.asarray(mapdl.mesh.nnum, dtype=int).copy()
    mapdl.allsel()
    index_by_node = {int(node): index for index, node in enumerate(nnum)}
    try:
        selected_indices = np.asarray([index_by_node[int(node)] for node in selected], dtype=int)
    except KeyError as exc:
        raise ThermalAnalysisError("Selected thermal boundary nodes do not match the solid mesh.") from exc
    if not len(selected_indices) or len(set(selected)) != len(selected):
        raise ThermalAnalysisError("MAPDL did not identify unique thermal boundary nodes.")
    selected_xy = np.asarray(xyz)[selected_indices, :2]
    tolerance = max(1e-10, float(mesh_size) * 1e-5)
    gauss, weights = np.polynomial.legendre.leggauss(5)
    u = (gauss + 1) / 2
    shape = np.column_stack(((1 - u) * (1 - 2 * u), 4 * u * (1 - u), u * (2 * u - 1)))
    quadrature = []
    covered = set()
    for segment, start in enumerate(load.xy):
        vector = load.xy[(segment + 1) % len(load.xy)] - start
        length = float(np.linalg.norm(vector))
        fraction = (selected_xy - start) @ vector / length**2
        distance = np.linalg.norm(selected_xy - (start + fraction[:, None] * vector), axis=1)
        mask = (distance <= tolerance) & (fraction >= -tolerance / length) & (fraction <= 1 + tolerance / length)
        positions = np.flatnonzero(mask)
        positions = positions[np.argsort(fraction[positions])]
        f = np.clip(fraction[positions], 0, 1)
        if len(f) < 3 or len(f) % 2 != 1 or not np.isclose(f[0], 0, atol=tolerance / length) or not np.isclose(f[-1], 1, atol=tolerance / length):
            raise ThermalAnalysisError("Cannot identify complete quadratic thermal boundary edges for energy integration.")
        for first in range(0, len(f) - 2, 2):
            left, middle, right = f[first:first + 3]
            if not np.isclose(middle, (left + right) / 2, rtol=1e-6, atol=tolerance / length):
                raise ThermalAnalysisError("Thermal boundary midside node is not on the expected straight quadratic edge.")
            indices = selected_indices[positions[first:first + 3]]
            covered.update(map(int, indices))
            quadrature.append((indices, shape.copy(), length * (right - left) * weights / 2,
                               segment, left + (right - left) * u))
    if covered != set(map(int, selected_indices)):
        raise ThermalAnalysisError("Boundary energy integration omitted selected FE nodes.")
    load.boundary_quadrature = quadrature
    return len(selected_indices)


def integrated_powers(load, temperatures, settings):
    """Integrate the same native line-interpolated H/Tdrive law over FE edges."""
    if not load.boundary_quadrature:
        raise ThermalAnalysisError("Missing FE boundary quadrature; surface energy cannot be reported as a global integral.")
    convective = radiative = 0.0
    for indices, shape, weights, segment, fraction in load.boundary_quadrature:
        next_index = (segment + 1) % len(load.xy)
        temperature = shape @ np.asarray(temperatures)[indices]
        h = (1 - fraction) * load.coefficient[segment] + fraction * load.coefficient[next_index]
        driving = (1 - fraction) * load.gas_temperature[segment] + fraction * load.gas_temperature[next_index]
        convective += float(weights @ (h * (driving - temperature)))
        radiative += float(weights @ (settings["emissivity"] * STEFAN_BOLTZMANN * (temperature**4 - settings["trad"]**4)))
    return np.asarray([convective, radiative, convective - radiative])
