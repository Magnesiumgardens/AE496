"""Constant or tabulated thermal properties with explicit material directions.

Tabulated heat capacity is integrated as specific sensible enthalpy (J/kg),
not replaced by a constant heat capacity in the stored-energy calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .thermal_loading import ThermalAnalysisError


def _positive_values(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise ThermalAnalysisError(f"{name} must contain positive finite numbers")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ThermalAnalysisError(f"{name} must contain positive finite numbers") from exc
    if array.ndim > 1 or not array.size or not np.isfinite(array).all() or (array <= 0).any():
        raise ThermalAnalysisError(f"{name} must contain positive finite numbers")
    return array


def _result(value):
    value = np.asarray(value)
    return float(value) if value.ndim == 0 else value


@dataclass(frozen=True)
class MaterialLaw:
    name: str
    density: float
    temperature_K: np.ndarray | None
    conductivity_axes: dict[str, np.ndarray]
    heat_capacity: np.ndarray
    extrapolation: str = "error"

    @property
    def is_tabulated(self):
        return self.temperature_K is not None

    @property
    def temperature_range_K(self):
        return None if self.temperature_K is None else (float(self.temperature_K[0]), float(self.temperature_K[-1]))

    def _temperatures(self, value, *, check_range=True):
        try:
            temperatures = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ThermalAnalysisError("Material evaluation requires positive finite temperatures in kelvin") from exc
        if not np.isfinite(temperatures).all() or (temperatures <= 0).any():
            raise ThermalAnalysisError("Material evaluation requires positive finite temperatures in kelvin")
        if check_range and self.temperature_K is not None and self.extrapolation == "error":
            if ((temperatures < self.temperature_K[0]) | (temperatures > self.temperature_K[-1])).any():
                low, high = self.temperature_range_K
                raise ThermalAnalysisError(f"Material temperature is outside the supplied {low:g}–{high:g} K table")
        return temperatures

    def _evaluate(self, temperatures, property_values):
        temperatures = self._temperatures(temperatures)
        if property_values.ndim == 0:
            return _result(np.full(temperatures.shape, float(property_values)))
        return _result(np.interp(temperatures, self.temperature_K, property_values))

    def specific_heat(self, temperature_K):
        """Return cp(T) in J/(kg K); endpoint extension is explicit policy."""
        return self._evaluate(temperature_K, self.heat_capacity)

    def conductivity(self, temperature_K, axis="x"):
        """Return conductivity along the named material direction, in W/(m K)."""
        direction = str(axis).lower()
        if direction not in self.conductivity_axes:
            raise ThermalAnalysisError(f"No conductivity supplied for material axis {axis!r}")
        return self._evaluate(temperature_K, self.conductivity_axes[direction])

    def _enthalpy_primitive(self, temperature_K):
        temperatures = self._temperatures(temperature_K)
        if self.heat_capacity.ndim == 0:
            return temperatures * float(self.heat_capacity)
        knots, capacity = self.temperature_K, self.heat_capacity
        slopes = np.diff(capacity) / np.diff(knots)
        cumulative = np.r_[0., np.cumsum(.5 * (capacity[:-1] + capacity[1:]) * np.diff(knots))]
        indices = np.clip(np.searchsorted(knots, temperatures, side="right") - 1, 0, len(knots) - 2)
        delta = np.clip(temperatures, knots[0], knots[-1]) - knots[indices]
        interior = cumulative[indices] + capacity[indices] * delta + .5 * slopes[indices] * delta**2
        return np.where(temperatures < knots[0], capacity[0] * (temperatures - knots[0]),
                        np.where(temperatures > knots[-1], cumulative[-1] + capacity[-1] * (temperatures - knots[-1]), interior))

    def enthalpy_change(self, temperature_K, reference_temperature_K=300.):
        """Exact integral of piecewise-linear cp(T), in J/kg relative to Tref."""
        return _result(self._enthalpy_primitive(temperature_K)
                       - self._enthalpy_primitive(reference_temperature_K))

    sensible_enthalpy = enthalpy_change

    def temperature_from_enthalpy(self, specific_enthalpy_J_kg, reference_temperature_K=300.):
        """Invert sensible enthalpy analytically; positive cp makes it monotone."""
        enthalpy = np.asarray(specific_enthalpy_J_kg, dtype=float)
        if not np.isfinite(enthalpy).all():
            raise ThermalAnalysisError("Sensible enthalpy must be finite")
        target = enthalpy + self._enthalpy_primitive(reference_temperature_K)
        if self.heat_capacity.ndim == 0:
            return _result(self._temperatures(target / float(self.heat_capacity)))
        knots, capacity = self.temperature_K, self.heat_capacity
        cumulative = np.r_[0., np.cumsum(.5 * (capacity[:-1] + capacity[1:]) * np.diff(knots))]
        if self.extrapolation == "error" and ((target < cumulative[0]) | (target > cumulative[-1])).any():
            raise ThermalAnalysisError("Sensible enthalpy is outside the supplied material temperature table")
        indices = np.clip(np.searchsorted(cumulative, target, side="right") - 1, 0, len(knots) - 2)
        change = np.clip(target, cumulative[0], cumulative[-1]) - cumulative[indices]
        slopes = np.diff(capacity) / np.diff(knots)
        # This rationalized root also handles a zero cp slope without division
        # by the slope and avoids subtracting nearly equal square roots.
        discriminant = capacity[indices]**2 + 2. * slopes[indices] * change
        delta = 2. * change / (capacity[indices] + np.sqrt(discriminant))
        interior = knots[indices] + delta
        temperatures = np.where(target < cumulative[0], knots[0] + target / capacity[0],
                                np.where(target > cumulative[-1], knots[-1] + (target - cumulative[-1]) / capacity[-1], interior))
        return _result(self._temperatures(temperatures))

    def range_diagnostics(self, temperature_K):
        """Describe extrapolation without changing or clipping temperature."""
        temperatures = self._temperatures(temperature_K, check_range=False)
        low, high = self.temperature_range_K or (None, None)
        below = int(np.count_nonzero(temperatures < low)) if low is not None else 0
        above = int(np.count_nonzero(temperatures > high)) if high is not None else 0
        return dict(status="outside_range" if below or above else "within_range" if self.is_tabulated else "constant_properties",
                    minimum_temperature_K=float(np.min(temperatures)), maximum_temperature_K=float(np.max(temperatures)),
                    table_minimum_K=low, table_maximum_K=high, below_table_count=below, above_table_count=above,
                    sample_count=int(temperatures.size), extrapolation=self.extrapolation)


def from_material_config(material):
    """Read SI properties with either temperature_k or temperature_c knots.

    Legacy scalar conductivity/cp configs remain valid. Conductivity may also
    be an isotropic table or an axis mapping, e.g. ``{x: [...], z: [...]}``.
    Tabulated entries share the supplied temperature grid; any scalar entry
    stays constant. Missing material directions are rejected when requested,
    never inferred from an unrelated axis.
    """
    if not isinstance(material, Mapping):
        raise ThermalAnalysisError("material must be a property mapping")
    temperature_keys = [key for key in ("temperature_k", "temperature_c") if key in material]
    if len(temperature_keys) > 1:
        raise ThermalAnalysisError("Specify material.temperature_k or material.temperature_c, not both")
    knots = None
    if temperature_keys:
        key = temperature_keys[0]
        try:
            knots = np.asarray(material[key], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ThermalAnalysisError("Material temperatures must be a strictly increasing finite table") from exc
        if key == "temperature_c":knots = knots + 273.15
        if knots.ndim != 1 or len(knots) < 2 or not np.isfinite(knots).all() or (knots <= 0).any() or (np.diff(knots) <= 0).any():
            raise ThermalAnalysisError("Material temperatures must be a strictly increasing finite positive kelvin table")
        knots = knots.copy()
        knots.setflags(write=False)
    extrapolation = str(material.get("extrapolation", "error"))
    if extrapolation not in {"error", "hold_endpoint"}:
        raise ThermalAnalysisError("Material extrapolation must be error or hold_endpoint")

    def property_values(name, value):
        values = _positive_values(name, value)
        if values.ndim == 1 and (knots is None or len(values) != len(knots)):
            raise ThermalAnalysisError(f"{name} must match the material temperature table")
        values = values.copy()
        values.setflags(write=False)
        return values

    density = _positive_values("material.density_kg_m3", material.get("density_kg_m3"))
    if density.ndim != 0:
        raise ThermalAnalysisError("Material density must be one constant value")
    capacity = property_values("material.specific_heat_j_kgk", material.get("specific_heat_j_kgk"))
    conductivity = material.get("conductivity_w_mk")
    if isinstance(conductivity, Mapping):
        axes = {str(axis).lower(): value for axis, value in conductivity.items()}
        if not axes or set(axes) - {"x", "y", "z"}:
            raise ThermalAnalysisError("Conductivity directions must be named x, y or z")
    else:
        axes = dict.fromkeys(("x", "y", "z"), conductivity)
    properties = {axis: property_values("material.conductivity_w_mk." + axis, value) for axis, value in axes.items()}
    return MaterialLaw(str(material.get("name", "user_material")), float(density), knots, properties, capacity, extrapolation)


def apply_material(mapdl, material, axes=("x", "y"), material_id=1):
    """Assign 2-D global conductivities from explicit orthotropic material axes.

    This is an axis mapping, not a claim that material axes follow a curved
    surface. MPTEMP/MPDATA carry at most six entries per command. Native MAPDL
    table extrapolation holds endpoint property values; ``error`` is enforced
    by post-solve range checks in the calling analysis.
    """
    axes = tuple(str(axis).lower() for axis in axes)
    if len(axes) != 2 or len(set(axes)) != 2 or any(axis not in material.conductivity_axes for axis in axes):
        raise ThermalAnalysisError("Select two distinct supplied material axes for the 2-D solid")
    if isinstance(material_id, bool) or int(material_id) != material_id or material_id <= 0:
        raise ThermalAnalysisError("Native material ID must be a positive integer")
    native = (("KXX", material.conductivity_axes[axes[0]]), ("KYY", material.conductivity_axes[axes[1]]),
              ("C", material.heat_capacity))
    mapdl.mp("DENS", int(material_id), material.density)
    if material.temperature_K is not None:
        if len(material.temperature_K) > 100:
            raise ThermalAnalysisError("Native MAPDL material tables support at most 100 temperature points")
        mapdl.mptemp()  # Clear prior abscissae before assigning this table.
        for first in range(0, len(material.temperature_K), 6):
            mapdl.mptemp(first + 1, *map(float, material.temperature_K[first:first + 6]))
    for label, values in native:
        if values.ndim == 0:
            mapdl.mp(label, int(material_id), float(values))
        else:
            for first in range(0, len(values), 6):
                mapdl.mpdata(label, int(material_id), first + 1, *map(float, values[first:first + 6]))
    return dict(material_name=material.name, density_kg_m3=material.density,
                global_xy_material_axes=list(axes), temperature_range_K=material.temperature_range_K,
                extrapolation=material.extrapolation, native_heat_capacity_property="C")
