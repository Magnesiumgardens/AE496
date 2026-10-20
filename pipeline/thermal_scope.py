"""Select a flat rear cut geometrically, independently of heating values."""
import numpy as np

from .thermal_loading import ThermalAnalysisError


def active_segments(xy, *, exclude_rear_face=False, nose_direction=(1., 0.), tolerance_fraction=1e-7):
    """Return the loaded outgoing edges of a closed ordered planar boundary.

    The rear is the single connected chain whose two edge endpoints lie on
    the minimum projection plane opposite the geometric nose direction. A
    face is never excluded merely because its calibration coefficient fails.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
        raise ThermalAnalysisError("Thermal scope requires a finite ordered XY polygon.")
    if not isinstance(exclude_rear_face, (bool, np.bool_)):
        raise ThermalAnalysisError("calibration.exclude_rear_face must be true or false.")
    if not exclude_rear_face:
        return np.ones(len(xy), dtype=bool)
    direction = np.asarray(nose_direction, dtype=float)
    if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ThermalAnalysisError("reporting.nose_direction must be a finite nonzero XY vector for rear selection.")
    direction = direction / np.linalg.norm(direction)
    try:
        tolerance_fraction = float(tolerance_fraction)
    except (ValueError, TypeError) as exc:
        raise ThermalAnalysisError("calibration.rear_plane_tolerance_fraction must be positive and below 0.01.") from exc
    if not np.isfinite(tolerance_fraction) or not 0 < tolerance_fraction < .01:
        raise ThermalAnalysisError("calibration.rear_plane_tolerance_fraction must be positive and below 0.01.")
    axial = xy @ direction
    span = float(np.ptp(axial))
    if span <= 1e-12:
        raise ThermalAnalysisError("Thermal rear selection has zero axial extent; check reporting.nose_direction.")
    rear_nodes = axial <= axial.min() + max(1e-12, tolerance_fraction * span)
    rear_edges = rear_nodes & np.roll(rear_nodes, -1)
    if not rear_edges.any() or rear_edges.all():
        raise ThermalAnalysisError("Cannot identify a flat rear closure; check reporting.nose_direction and the rear-plane tolerance.")
    if np.count_nonzero(rear_edges != np.roll(rear_edges, 1)) != 2:
        raise ThermalAnalysisError("Rear closure is not a single connected edge chain; thermal scope is ambiguous.")
    active = ~rear_edges
    if np.count_nonzero(active) < 2:
        raise ThermalAnalysisError("Rear exclusion must retain at least two external surface edges.")
    return active


def active_nodes(segments):
    """A junction belongs to the heated surface if either adjacent edge does."""
    segments = np.asarray(segments, dtype=bool)
    return segments | np.roll(segments, 1)


def scoped_face_values_to_nodes(values, lengths, active):
    """Project only selected adjacent faces, without contamination at corners.

    Zeros at rear-only vertices are explicitly inactive placeholders. They
    are never used as a calibrated coefficient or prescribed heat flux.
    """
    values, lengths, active = np.asarray(values, dtype=float), np.asarray(lengths, dtype=float), np.asarray(active, dtype=bool)
    if values.shape != lengths.shape or values.shape != active.shape or values.ndim != 1:
        raise ThermalAnalysisError("Thermal projection arrays do not match the closed boundary.")
    if not np.isfinite(lengths).all() or (lengths <= 0).any() or not np.isfinite(values[active]).all():
        raise ThermalAnalysisError("Thermal projection requires positive lengths and finite active face values.")
    weights = np.where(active, lengths, 0.)
    weighted = np.where(active, values, 0.) * weights
    denominator = weights + np.roll(weights, 1)
    return np.divide(weighted + np.roll(weighted, 1), denominator,
                     out=np.zeros_like(weighted), where=denominator > 0)
