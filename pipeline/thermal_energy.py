"""Check solved solid thermal energy against the applied surface-load laws.

The stored-energy integral uses the actual six-node solid triangles, including
their midside temperatures.  It is independent of both nodal arithmetic means
and the boundary-power integral.  No surface-effect elements or ambient nodes
belong in this quadrature.  All integrals use a unit out-of-plane span.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .thermal_loading import ThermalAnalysisError


@dataclass(frozen=True)
class SolidEnergyQuadrature:
    node_numbers: np.ndarray
    nodal_volume_weights_m2: np.ndarray
    area_m2: float
    element_count: int

    def stored_energy_change(self, temperatures, initial_temperature, density, specific_heat):
        """Return FE-integrated sensible energy relative to a uniform initial T."""
        values = np.asarray(temperatures, dtype=float)
        if values.shape[-1:] != (len(self.node_numbers),) or not np.isfinite(values).all():
            raise ThermalAnalysisError("Solid temperatures do not match the stored-energy mesh.")
        return density * specific_heat * ((values - initial_temperature) @ self.nodal_volume_weights_m2)


def normalize_solid_connectivity(nnum, xyz, raw_records):
    """Normalize pre-surface-element MAPDL records into straight triangles.

    ``mapdl.mesh.elem`` records have ten header fields followed by MAPDL node
    numbers. PLANE293's triangular form can appear as a degenerate eight-node
    connectivity, so retaining the first six entries is unsafe. Identify the
    three corners by maximum enclosed area, then verify the remaining unique
    nodes at the three side midpoints. This runner creates straight triangles;
    an unexpected curved/malformed record is rejected, never silently remapped.

    The snapshot must be captured before adding SURF151 elements: all element
    records must be live material-1/type-1 solids. Surface nodes and elements
    must not acquire artificial solid heat capacity in the energy check.
    """
    numbers = np.asarray(nnum)
    coordinates = np.asarray(xyz, dtype=float)
    if (numbers.ndim != 1 or not np.issubdtype(numbers.dtype, np.integer)
            or len(np.unique(numbers)) != len(numbers)
            or coordinates.ndim != 2 or coordinates.shape[1] < 2 or len(coordinates) != len(numbers)
            or not np.isfinite(coordinates).all()):
        raise ThermalAnalysisError("Invalid solid node snapshot for native connectivity normalization.")
    index_by_number = {int(node): index for index, node in enumerate(numbers)}
    normalized = []
    try:
        records = list(raw_records)
    except TypeError as exc:
        raise ThermalAnalysisError("Native solid mesh connectivity is unavailable.") from exc
    if not records:
        raise ThermalAnalysisError("Native solid mesh connectivity is empty.")
    for record in records:
        record = np.asarray(record)
        if (record.ndim != 1 or len(record) < 16 or not np.issubdtype(record.dtype, np.integer)
                or record[0] != 1 or record[1] != 1 or record[5] != 0):
            raise ThermalAnalysisError("Energy connectivity must contain only live material-1/type-1 solid elements.")
        if (record[10:] < 0).any():
            raise ThermalAnalysisError("Native solid connectivity contains invalid negative node numbers.")
        unique = list(dict.fromkeys(int(node) for node in record[10:] if node > 0))
        if len(unique) != 6:
            raise ThermalAnalysisError("A native solid element does not contain exactly six distinct thermal nodes.")
        try:
            xy = coordinates[[index_by_number[node] for node in unique], :2]
        except KeyError as exc:
            raise ThermalAnalysisError("Native solid connectivity refers to an unknown mesh node.") from exc
        xy = xy - xy[0]
        characteristic = float(np.ptp(xy, axis=0).max())
        tolerance = max(1e-12, characteristic * 1e-7)
        def area2(indices):
            a, b, c = xy[list(indices)]
            first, second = b - a, c - a
            return first[0] * second[1] - first[1] * second[0]
        corners = list(max(combinations(range(6), 3), key=lambda indices: abs(area2(indices))))
        if abs(area2(corners)) <= max(1e-30, characteristic**2 * 1e-12):
            raise ThermalAnalysisError("Native solid connectivity contains a degenerate triangle.")
        if area2(corners) < 0:
            corners[1], corners[2] = corners[2], corners[1]
        remaining = set(range(6)) - set(corners)
        midsides = []
        for first, second in zip(corners, corners[1:] + corners[:1]):
            midpoint = .5 * (xy[first] + xy[second])
            candidates = [index for index in remaining if np.linalg.norm(xy[index] - midpoint) <= tolerance]
            if len(candidates) != 1:
                raise ThermalAnalysisError("Native solid triangle does not have uniquely matching straight-edge midside nodes.")
            midsides.append(candidates[0])
            remaining.remove(candidates[0])
        normalized.append([unique[index] for index in corners + midsides])
    return np.asarray(normalized, dtype=int)


def _triangle_rule():
    """Tensor Gauss integration mapped onto the unit reference triangle.

    Four points in each coordinate integrate the quadratic shape functions
    times the quadratic isoparametric Jacobian exactly, including curved
    six-node triangles with a nonfolded quadratic geometry map.
    """
    points, weights = np.polynomial.legendre.leggauss(4)
    points, weights = (points + 1.) / 2., weights / 2.
    u, v = np.meshgrid(points, points, indexing="ij")
    wu, wv = np.meshgrid(weights, weights, indexing="ij")
    r, s = u.ravel(), ((1. - u) * v).ravel()
    integration_weights = (wu * wv * (1. - u)).ravel()
    first, second, third = 1. - r - s, r, s
    shape = np.column_stack((first * (2. * first - 1.), second * (2. * second - 1.),
                             third * (2. * third - 1.), 4. * first * second,
                             4. * second * third, 4. * third * first))
    dr = np.column_stack((1. - 4. * first, 4. * second - 1., np.zeros_like(r),
                          4. * (first - second), 4. * third, -4. * third))
    ds = np.column_stack((1. - 4. * first, np.zeros_like(r), 4. * third - 1.,
                          -4. * second, 4. * second, 4. * (first - third)))
    return shape, dr, ds, integration_weights


def prepare_solid_energy(nnum, xyz, triangle_node_ids):
    """Build energy weights from normalized six-node triangle connectivity.

    ``triangle_node_ids`` has one row per *solid* element and uses node numbers
    ordered ``I,J,K,mid_IJ,mid_JK,mid_KI``.  The VTK quadratic-triangle cell
    representation supplies this order; raw degenerate MAPDL eight-node
    connectivity must first be normalized by the native mesh reader.
    """
    numbers = np.asarray(nnum)
    coordinates = np.asarray(xyz, dtype=float)
    triangles = np.asarray(triangle_node_ids)
    if (numbers.ndim != 1 or not len(numbers) or not np.issubdtype(numbers.dtype, np.integer)
            or (numbers <= 0).any() or len(np.unique(numbers)) != len(numbers)):
        raise ThermalAnalysisError("Stored-energy integration requires unique positive solid node numbers.")
    if (coordinates.ndim != 2 or len(coordinates) != len(numbers) or coordinates.shape[1] < 2
            or not np.isfinite(coordinates).all()):
        raise ThermalAnalysisError("Stored-energy integration requires finite solid-node coordinates.")
    if (triangles.ndim != 2 or triangles.shape[1] != 6 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer)):
        raise ThermalAnalysisError("Stored-energy integration requires six-node solid triangles only.")
    index_by_number = {int(node): index for index, node in enumerate(numbers)}
    shape, dr, ds, reference_weights = _triangle_rule()
    volume_weights = np.zeros(len(numbers), dtype=float)
    seen = set()
    area = 0.
    for triangle in triangles:
        identity = tuple(sorted(map(int, triangle)))
        if len(set(identity)) != 6 or identity in seen:
            raise ThermalAnalysisError("Stored-energy connectivity contains duplicate or repeated triangle nodes.")
        seen.add(identity)
        try:
            indices = np.asarray([index_by_number[int(node)] for node in triangle], dtype=int)
        except KeyError as exc:
            raise ThermalAnalysisError("A stored-energy element refers to a node outside the solid mesh.") from exc
        xy = coordinates[indices, :2]
        # Centering avoids cancellation for a small nose far from the origin.
        xy = xy - xy[0]
        first, second = dr @ xy, ds @ xy
        determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        tolerance = max(1e-30, float(np.ptp(xy, axis=0).max())**2 * 1e-12)
        if (np.abs(determinant) <= tolerance).any() or not ((determinant > 0).all() or (determinant < 0).all()):
            raise ThermalAnalysisError("Stored-energy integration found a degenerate or folded solid triangle.")
        weights = reference_weights * np.abs(determinant)
        area += float(np.sum(weights))
        np.add.at(volume_weights, indices, weights @ shape)
    if not np.isfinite(area) or area <= 0 or not np.isclose(volume_weights.sum(), area, rtol=1e-11, atol=1e-20):
        raise ThermalAnalysisError("Stored-energy volume weights do not reproduce the solid area.")
    return SolidEnergyQuadrature(numbers.copy(), volume_weights, area, len(triangles))


def evaluate_energy_balance(quadrature, times, temperatures, powers, settings,
                            relative_tolerance=.02, absolute_tolerance_J_m=1e-3):
    """Compare solved stored energy to every native backward-Euler substep.

    Histories include the uniform initial state at ``times[0] == 0`` and every
    native solved substep afterward. ``powers`` columns are convection in,
    radiation out, and their difference in W/m of span.  The backward-Euler
    endpoint integral is the gate because MAPDL is explicitly configured with
    THETA=1. The trapezoidal integral is retained for the presentation history
    and its difference is diagnostic, not a second contradictory solver gate.

    Tolerances scale with gross incoming-plus-outgoing energy rather than the
    almost-cancelled net energy near equilibrium.  Step and cumulative checks
    are both retained; neither a final-only match nor a fitted correction can
    conceal an earlier energy-balance failure.
    """
    times = np.asarray(times, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    powers = np.asarray(powers, dtype=float)
    try:
        density, heat_capacity, initial = (float(settings[key]) for key in ("rho", "cp", "initial_t"))
        relative_tolerance, absolute_tolerance_J_m = float(relative_tolerance), float(absolute_tolerance_J_m)
    except (KeyError, TypeError, ValueError) as exc:
        raise ThermalAnalysisError("Energy verification requires constant positive density, heat capacity and initial temperature.") from exc
    if (not np.isfinite([density, heat_capacity, initial, relative_tolerance, absolute_tolerance_J_m]).all()
            or min(density, heat_capacity, initial, absolute_tolerance_J_m) <= 0
            or relative_tolerance <= 0 or relative_tolerance >= 1):
        raise ThermalAnalysisError("Invalid material properties or tolerances for stored-energy verification.")
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or abs(times[0]) > 1e-12 or (np.diff(times) <= 0).any()):
        raise ThermalAnalysisError("Energy verification requires physical time zero and every increasing solved substep.")
    if (temperatures.shape != (len(times), len(quadrature.node_numbers))
            or not np.isfinite(temperatures).all() or (temperatures <= 0).any()
            or not np.allclose(temperatures[0], initial, rtol=0, atol=1e-8)):
        raise ThermalAnalysisError("Energy verification requires solved solid temperatures and the uniform configured initial state.")
    if (powers.shape != (len(times), 3) or not np.isfinite(powers).all()
            or not np.allclose(powers[:, 2], powers[:, 0] - powers[:, 1], rtol=1e-10, atol=1e-8)):
        raise ThermalAnalysisError("Energy verification requires finite, consistent convection/radiation/net surface powers.")

    stored = quadrature.stored_energy_change(temperatures, initial, density, heat_capacity)
    dt = np.diff(times)
    implicit_steps = dt[:, None] * powers[1:]
    trapezoidal_steps = dt[:, None] * .5 * (powers[1:] + powers[:-1])
    implicit = np.vstack((np.zeros(3), np.cumsum(implicit_steps, axis=0)))
    trapezoidal = np.vstack((np.zeros(3), np.cumsum(trapezoidal_steps, axis=0)))
    stored_steps = np.diff(stored)
    gross_steps = dt * (np.abs(powers[1:, 0]) + np.abs(powers[1:, 1]))
    step_tolerance = absolute_tolerance_J_m + relative_tolerance * gross_steps
    cumulative_tolerance = absolute_tolerance_J_m + relative_tolerance * np.cumsum(gross_steps)
    step_errors = stored_steps - implicit_steps[:, 2]
    cumulative_errors = stored[1:] - implicit[1:, 2]
    step_passed = np.abs(step_errors) <= step_tolerance
    cumulative_passed = np.abs(cumulative_errors) <= cumulative_tolerance
    # Only count a wrong sign when both terms exceed the numerical tolerance.
    # Tiny sign changes near equilibrium are not evidence of a broken model.
    sign_passed = ~((stored[1:] * implicit[1:, 2] < 0)
                    & (np.abs(stored[1:]) > cumulative_tolerance)
                    & (np.abs(implicit[1:, 2]) > cumulative_tolerance))
    rows = []
    for index, time in enumerate(times):
        rows.append({
            "time_s": float(time),
            "solid_internal_energy_change_per_unit_span_J_m": float(stored[index]),
            "cumulative_implicit_convective_energy_per_unit_span_J_m": float(implicit[index, 0]),
            "cumulative_implicit_radiative_energy_per_unit_span_J_m": float(implicit[index, 1]),
            "cumulative_implicit_net_energy_per_unit_span_J_m": float(implicit[index, 2]),
            "cumulative_trapezoidal_net_energy_per_unit_span_J_m": float(trapezoidal[index, 2]),
            "step_energy_balance_error_J_m": float(step_errors[index - 1]) if index else 0.,
            "cumulative_energy_balance_error_J_m": float(cumulative_errors[index - 1]) if index else 0.,
            "step_energy_balance_tolerance_J_m": float(step_tolerance[index - 1]) if index else absolute_tolerance_J_m,
            "cumulative_energy_balance_tolerance_J_m": float(cumulative_tolerance[index - 1]) if index else absolute_tolerance_J_m,
            "step_energy_balance_passed": bool(step_passed[index - 1]) if index else True,
            "cumulative_energy_balance_passed": bool(cumulative_passed[index - 1]) if index else True,
            "energy_sign_passed": bool(sign_passed[index - 1]) if index else True,
        })
    worst_index = int(np.argmax(np.abs(step_errors) / step_tolerance))
    passed = bool(step_passed.all() and cumulative_passed.all() and sign_passed.all())
    summary = {
        "status": "passed" if passed else "failed",
        "method": "FE-integrated solid sensible energy versus independently integrated surface-load laws",
        "stored_energy_integration": "Six-node isoparametric solid triangles; all corner and midside temperatures; unit span",
        "time_integration": "Backward Euler (THETA=1), endpoint power at every native solved substep",
        "trapezoidal_interpretation": "Presentation integral only; not used to gate the backward-Euler solver balance",
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance_J_m": absolute_tolerance_J_m,
        "tolerance_scale": "Absolute convection plus absolute radiation energy (gross exchange)",
        "solid_area_m2": float(quadrature.area_m2),
        "solid_mass_per_unit_span_kg_m": float(density * quadrature.area_m2),
        "solid_element_count": int(quadrature.element_count),
        "solved_substeps_checked": int(len(times) - 1),
        "all_steps_passed": bool(step_passed.all()),
        "all_cumulative_steps_passed": bool(cumulative_passed.all()),
        "energy_sign_passed": bool(sign_passed.all()),
        "final_balance_passed": bool(cumulative_passed[-1] and sign_passed[-1]),
        "final_stored_energy_change_per_unit_span_J_m": float(stored[-1]),
        "final_implicit_net_energy_per_unit_span_J_m": float(implicit[-1, 2]),
        "final_trapezoidal_net_energy_per_unit_span_J_m": float(trapezoidal[-1, 2]),
        "final_energy_balance_error_J_m": float(cumulative_errors[-1]),
        "maximum_absolute_step_error_J_m": float(np.max(np.abs(step_errors))),
        "maximum_absolute_cumulative_error_J_m": float(np.max(np.abs(cumulative_errors))),
        "maximum_step_error_to_tolerance": float(np.max(np.abs(step_errors) / step_tolerance)),
        "worst_step_time_s": float(times[worst_index + 1]),
        "native_internal_energy_balance_verified": passed,
    }
    return rows, summary
