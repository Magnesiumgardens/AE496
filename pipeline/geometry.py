"""Label a prepared XY fluid surface using its actual CAD topology."""
from __future__ import annotations

import math
from pathlib import Path

from .utils import work_paths


class GeometryPipelineError(RuntimeError):
    pass


def _all_bodies(component):
    """Designs are components; CAD imports can nest bodies in assemblies."""
    yield from component.bodies
    for child in component.components:
        yield from _all_bodies(child)


def _metres(value) -> float:
    # Point3D coordinates are quantities, not bare coordinates in CAD units.
    return float(value.to("m").magnitude)


def _boundary_edges(design, tolerance_m: float = 1.0e-8):
    """Return (farfield edges, body edges), rejecting ambiguous CAD inputs."""
    bodies = list(_all_bodies(design))
    if len(bodies) != 1 or not bodies[0].is_surface:
        raise GeometryPipelineError(
            "geometry.input_kind=prepared_fluid_surface requires exactly one "
            "surface body: the farfield rectangle with the vehicle removed."
        )
    faces = list(bodies[0].faces)
    if len(faces) != 1:
        raise GeometryPipelineError("The fluid surface must have exactly one face.")
    face = faces[0]
    if getattr(face.surface_type, "name", "") != "SURFACETYPE_PLANE":
        raise GeometryPipelineError("The fluid face must be planar in the global XY plane.")
    loops = list(face.loops)
    if len(loops) != 2:
        raise GeometryPipelineError(
            "The fluid face must have one outer farfield loop and one inner vehicle loop."
        )
    by_type = {}
    for loop in loops:
        loop_type = getattr(loop.type, "name", "")
        if loop_type not in {"OUTER_LOOP", "INNER_LOOP"} or loop_type in by_type:
            raise GeometryPipelineError("CAD must expose one OUTER_LOOP and one INNER_LOOP.")
        # Loop bounding boxes include the entire curved edges. Face/edge vertex APIs
        # in Geometry 0.17.2 require a newer backend than the working 25.1 backend.
        for corner in (loop.min_bbox, loop.max_bbox):
            z = _metres(corner.z)
            if not math.isfinite(z) or abs(z) > tolerance_m:
                raise GeometryPipelineError(
                    f"The fluid surface must be at z=0 in metres; loop bound z={z:g}."
                )
        edges = list(loop.edges)
        if not edges:
            raise GeometryPipelineError("CAD returned a boundary loop without edges.")
        by_type[loop_type] = edges
    return by_type["OUTER_LOOP"], by_type["INNER_LOOP"]


def _label_boundaries(design, farfield, body, cfg):
    names = {
        cfg.get("farfield_name", "farfield"): farfield,
        cfg.get("body_wall_name", "body_wall"): body,
    }
    if len(names) != 2:
        raise GeometryPipelineError("Farfield and body wall named selections must differ.")
    selections = {selection.name: selection for selection in design.named_selections}
    if cfg.get("auto_label", True):
        for name, edges in names.items():
            if name in selections:
                design.delete_named_selection(selections[name])
            design.create_named_selection(name=name, edges=edges)
    else:
        for name, edges in names.items():
            existing = selections.get(name)
            if existing is None or {edge.id for edge in existing.edges} != {
                edge.id for edge in edges
            }:
                raise GeometryPipelineError(
                    f"auto_label=false requires named selection {name!r} to match its CAD loop."
                )


def prepare_geometry(cfg: dict) -> Path:
    """Launch the known-working SpaceClaim backend and return its exported PMDB."""
    geometry = cfg["geometry"]
    source = Path(geometry["input_file"]).resolve()
    if not source.is_file():
        raise GeometryPipelineError(f"Geometry input does not exist: {source}")
    if geometry.get("input_kind", "prepared_fluid_surface") != "prepared_fluid_surface":
        raise GeometryPipelineError("Only geometry.input_kind=prepared_fluid_surface is supported.")
    tolerance = float(geometry.get("planar_tolerance_m", 1.0e-8))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise GeometryPipelineError("geometry.planar_tolerance_m must be finite and >0.")
    runtime = cfg.get("runtime", {})
    mode = runtime.get("geometry_launch_mode", "spaceclaim")
    if mode == "auto":
        raise GeometryPipelineError(
            "Select runtime.geometry_launch_mode explicitly; SpaceClaim is the "
            "known-working backend for this installation."
        )
    try:
        from ansys.geometry.core import launch_modeler
    except ImportError as exc:
        raise GeometryPipelineError("Install the project's PyAnsys Geometry dependencies.") from exc
    paths = work_paths(cfg)
    export_dir = paths["labeled_geometry"]
    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"[geometry] Launching PyAnsys Geometry ({mode})...")
    modeler = None
    try:
        modeler = launch_modeler(
            mode=mode,
            version=int(runtime.get("geometry_version", 251)),
            timeout=int(runtime.get("geometry_start_timeout_s", 300)),
        )
        print(f"[geometry] Reading prepared fluid surface: {source}")
        design = modeler.open_file(str(source))
        farfield, body = _boundary_edges(design, tolerance)
        _label_boundaries(design, farfield, body, geometry)
        # Despite the argument name, Geometry 0.17.2 treats location as a
        # DIRECTORY and appends design.name + '.pmdb'. Use the returned path.
        exported = Path(design.export_to_pmdb(location=export_dir)).resolve()
        if not exported.is_file() or exported.stat().st_size == 0:
            raise GeometryPipelineError(f"Geometry export did not create a nonempty PMDB: {exported}")
        print(f"[geometry] Labeled PMDB: {exported}")
        return exported
    except GeometryPipelineError:
        raise
    except Exception as exc:
        raise GeometryPipelineError(f"Geometry preparation failed: {exc}") from exc
    finally:
        if modeler is not None:
            try:
                modeler.close()
            except Exception as exc:
                print(f"[geometry] Warning: owned modeler could not close: {exc}")
