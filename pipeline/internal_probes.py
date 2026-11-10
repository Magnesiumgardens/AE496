"""Geometry-defined internal probes and exact quadratic FE interpolation.

Depths and coordinates are metres. Axial distance behind a nose tip and inward
distance from an exposed surface are deliberately separate conventions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .thermal_loading import ThermalAnalysisError, positive


def _polygon(boundary_xy):
    xy = np.asarray(boundary_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
        raise ThermalAnalysisError("Internal probes require a finite ordered solid polygon")
    edges = np.roll(xy, -1, axis=0) - xy
    scale = float(np.ptp(xy, axis=0).max())
    tolerance = max(1e-12, scale * 1e-10)
    if (np.linalg.norm(edges, axis=1) <= tolerance).any():
        raise ThermalAnalysisError("Internal-probe polygon contains duplicate or zero-length edges")
    centered = xy - xy[0]
    area2 = float(np.sum(centered[:, 0] * np.roll(centered[:, 1], -1)
                         - centered[:, 1] * np.roll(centered[:, 0], -1)))
    if abs(area2) <= max(1e-30, scale**2 * 1e-12):
        raise ThermalAnalysisError("Internal-probe polygon has zero area")
    return xy, edges, area2, tolerance


def _distances(point, starts, edges):
    lengths2 = np.sum(edges**2, axis=1)
    fractions = np.clip(np.sum((point - starts) * edges, axis=1) / lengths2, 0., 1.)
    return np.linalg.norm(point - (starts + fractions[:, None] * edges), axis=1)


def _strictly_inside(point, xy, edges, tolerance):
    if float(np.min(_distances(point, xy, edges))) <= tolerance:
        return False
    crossings = 0
    x, y = point
    for start, end in zip(xy, np.roll(xy, -1, axis=0)):
        if (start[1] > y) != (end[1] > y):
            cross_x = start[0] + (y - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
            crossings += cross_x > x
    return bool(crossings % 2)


def axial_tip_probe(boundary_xy, depth_m, nose_direction=(1., 0.)):
    """Return the point ``depth_m`` behind the tip along the configured axis.

    A flat tip uses its transverse midpoint, independent of edge sampling.
    This is an axial offset, not a distance from every exposed surface.
    """
    depth = positive("internal probe depth_m", depth_m)
    xy, edges, _, tolerance = _polygon(boundary_xy)
    direction = np.asarray(nose_direction, dtype=float)
    if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ThermalAnalysisError("Internal probe nose_direction must be a finite nonzero XY vector")
    direction = direction / np.linalg.norm(direction)
    transverse = np.asarray([-direction[1], direction[0]])
    relative = xy - xy[0]
    axial = relative @ direction
    forward = relative[axial >= axial.max() - tolerance]
    width_coordinates = forward @ transverse
    tip = xy[0] + axial.max() * direction + .5 * (width_coordinates.min() + width_coordinates.max()) * transverse
    probe = tip - depth * direction
    if not _strictly_inside(probe, xy, edges, tolerance):
        raise ThermalAnalysisError("Axial internal probe falls outside or on the solid boundary; no nearest-node substitution was made")
    return probe


def surface_normal_probes(boundary_xy, active_segments, depth_m, *, require_min_clearance=False):
    """Offset each exposed edge midpoint inward by exactly the requested depth.

    The nearest *other* heated surface may be closer than the source edge.
    Every accepted point reports that distance. With ``require_min_clearance``
    enabled such points are excluded, rather than called equally protected.
    Zero valid probes is returned explicitly for a solid that is too thin.
    The input is an ordered, non-self-intersecting solid polygon.
    """
    depth = positive("internal probe depth_m", depth_m)
    xy, edges, area2, tolerance = _polygon(boundary_xy)
    active = np.asarray(active_segments)
    if active.shape != (len(xy),) or not np.isin(active, (0, 1)).all() or not active.any():
        raise ThermalAnalysisError("Internal probes require a nonempty active surface-segment mask")
    active = active.astype(bool)
    lengths = np.linalg.norm(edges, axis=1)
    normals = np.column_stack((-edges[:, 1], edges[:, 0])) / lengths[:, None]
    if area2 < 0:normals *= -1
    probes, excluded = [], []
    for index in np.flatnonzero(active):
        origin = xy[index] + .5 * edges[index]
        point = origin + depth * normals[index]
        closest = float(np.min(_distances(point, xy[active], edges[active])))
        row = dict(probe_id=f"surface_normal_{index}", source_segment=int(index), depth_m=depth,
                   x_m=float(point[0]), y_m=float(point[1]),
                   source_x_m=float(origin[0]), source_y_m=float(origin[1]),
                   inward_normal_x=float(normals[index, 0]), inward_normal_y=float(normals[index, 1]),
                   nearest_heated_surface_distance_m=closest,
                   minimum_heated_surface_clearance_satisfied=closest >= depth - tolerance)
        if not _strictly_inside(point, xy, edges, tolerance):
            excluded.append(dict(row, reason="surface-normal offset lies outside or on the solid boundary"))
        elif require_min_clearance and closest < depth - tolerance:
            excluded.append(dict(row, reason="another heated surface is closer than the required depth"))
        else:
            probes.append(row)
    return dict(status="ready" if probes else "unavailable",
                points=np.asarray([[row["x_m"], row["y_m"]] for row in probes], dtype=float).reshape(-1, 2),
                probes=probes, excluded=excluded)


@dataclass(frozen=True)
class ProbeInterpolation:
    node_indices: np.ndarray
    shape_weights: np.ndarray
    node_count: int
    probe_points: np.ndarray
    triangle_indices: np.ndarray

    def evaluate(self, temperatures):
        values = np.asarray(temperatures, dtype=float)
        if values.ndim < 1 or values.shape[-1] != self.node_count or not np.isfinite(values).all() or (values <= 0).any():
            raise ThermalAnalysisError("Probe temperatures must match the finite positive solid nodal field")
        sampled = np.einsum("...pk,pk->...p", values[..., self.node_indices], self.shape_weights)
        if not np.isfinite(sampled).all() or (sampled <= 0).any():
            raise ThermalAnalysisError("Quadratic interpolation returned an invalid absolute probe temperature")
        return sampled


def prepare_probe_interpolation(nnum, xyz, triangles, probe_points):
    """Locate probes in straight six-node triangles and retain FE shape weights.

    Connectivity is node IDs in ``I,J,K,mid_IJ,mid_JK,mid_KI`` order. Both
    boundary points and interior points can be evaluated. Outside-mesh points
    are rejected; the implementation never samples a nearby wall/node instead.
    """
    numbers = np.asarray(nnum)
    coordinates, cells, points = np.asarray(xyz, dtype=float), np.asarray(triangles), np.asarray(probe_points, dtype=float)
    if (numbers.ndim != 1 or not len(numbers) or not np.issubdtype(numbers.dtype, np.integer)
            or (numbers <= 0).any() or len(set(numbers)) != len(numbers)
            or coordinates.ndim != 2 or coordinates.shape[0] != len(numbers) or coordinates.shape[1] < 2
            or not np.isfinite(coordinates).all()):
        raise ThermalAnalysisError("Invalid solid node IDs or coordinates for internal probes")
    if cells.ndim != 2 or cells.shape[1] != 6 or not len(cells) or not np.issubdtype(cells.dtype, np.integer):
        raise ThermalAnalysisError("Internal probes require normalized six-node solid triangles")
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ThermalAnalysisError("Internal probes require at least one finite XY location")
    by_id = {int(node): index for index, node in enumerate(numbers)}
    try:
        indices = np.asarray([[by_id[int(node)] for node in cell] for cell in cells], dtype=int)
    except KeyError as exc:
        raise ThermalAnalysisError("Internal-probe connectivity refers to a node outside the solid mesh") from exc
    if any(len(set(cell)) != 6 for cell in cells) or len({tuple(sorted(cell)) for cell in cells}) != len(cells):
        raise ThermalAnalysisError("Internal-probe mesh contains repeated/duplicate triangle nodes")
    mesh_xy = coordinates[indices, :2]
    first = mesh_xy[:, 1] - mesh_xy[:, 0]
    second = mesh_xy[:, 2] - mesh_xy[:, 0]
    determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    scales = np.ptp(mesh_xy, axis=1).max(axis=1)
    if (np.abs(determinant) <= np.maximum(1e-30, scales**2 * 1e-12)).any():
        raise ThermalAnalysisError("Internal-probe mesh contains a degenerate triangle")
    midpoints = .5 * (mesh_xy[:, [0, 1, 2]] + mesh_xy[:, [1, 2, 0]])
    if (np.linalg.norm(mesh_xy[:, 3:] - midpoints, axis=2) > np.maximum(1e-12, scales[:, None] * 1e-7)).any():
        raise ThermalAnalysisError("Internal-probe interpolation requires straight quadratic triangle edges")
    transforms = np.linalg.inv(np.stack((first, second), axis=2))
    chosen, weights = [], []
    for point in points:
        local = np.einsum("nij,nj->ni", transforms, point - mesh_xy[:, 0])
        barycentric = np.column_stack((1. - local.sum(axis=1), local))
        eligible = np.flatnonzero((barycentric >= -1e-10).all(axis=1) & (barycentric <= 1. + 1e-10).all(axis=1))
        if not len(eligible):
            raise ThermalAnalysisError(f"Internal probe {point.tolist()} lies outside the solid FE mesh; no nearest-node substitution was made")
        triangle = int(eligible[0])
        a, b, c = barycentric[triangle]
        chosen.append(triangle)
        weights.append([a * (2*a - 1), b * (2*b - 1), c * (2*c - 1), 4*a*b, 4*b*c, 4*c*a])
    return ProbeInterpolation(indices[chosen], np.asarray(weights), len(numbers), points.copy(), np.asarray(chosen))
