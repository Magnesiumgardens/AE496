"""The 2025 R1 two-dimensional Fluent meshing workflow."""
from __future__ import annotations

from pathlib import Path
import uuid

from .utils import work_paths


class MeshPipelineError(RuntimeError):
    def __init__(self, message: str, *, stage: str = "preflight"):
        super().__init__(message)
        self.stage = stage


def _workflow_task(workflow, *names):
    """Resolve release-specific names within the 2-D workflow only."""
    for name in names:
        try:
            return getattr(workflow, name)
        except AttributeError:
            continue
    raise MeshPipelineError(
        "None of the expected 2-D workflow tasks are available: " + ", ".join(names)
    )


def _run_task(workflow, names, state=None):
    task = _workflow_task(workflow, *names)
    if state:
        task.arguments.update_dict(state)
    if task() is False:
        raise MeshPipelineError(f"Fluent reported failure for the 2-D task {names[0]}.")
    return task


def _make_curvature_sizing(workflow, label: str, cfg: dict):
    task = _workflow_task(workflow, "add_local_sizing_2d", "add_local_sizing_wtm")
    mesh = cfg["mesh"]
    child = task.add_child_and_update(
        state={
            "boi_control_name": "body_curvature",
            "boi_execution": "Curvature",
            "boi_min_size": float(mesh["body_min_size"]),
            "boi_max_size": float(mesh["body_max_size"]),
            "boi_curvature_normal_angle": float(mesh.get("body_curvature_normal_angle_deg", 8.0)),
            "boi_scope_to": "edges",
            "boi_zoneor_label": "label",
            "edge_label_list": [label],
            "draw_size_control": False,
        },
        defer_update=False,
    )
    # add_child_and_update executes the inserted compound child. Do not execute
    # the parent again (which can add a second control or reset the selection).
    return child


def _add_boundary_layers(workflow, label: str, settings: dict):
    task = _workflow_task(workflow, "add_2d_boundary_layers")
    state = {
            "bl_control_name": "body_boundary_layers",
            "offset_method_type": settings.get("offset_method", "last-ratio"),
            "number_of_layers": int(settings.get("number_of_layers", 20)),
            "first_layer_height": float(settings.get("first_layer_height", 1.0e-5)),
            "rate": float(settings.get("growth_rate", 1.15)),
            "edge_label_list": [label],
    }
    # GrowOn values are supplied by Fluent and are not documented in the pinned
    # Python schema. Scope to the explicit body label, using a supported enum
    # when available. There is exactly one wall loop; the farfield is not a wall.
    try:
        allowed = list(task.arguments.grow_on.allowed_values() or [])
    except AttributeError:
        allowed = []
    requested = settings.get("grow_on")
    if requested is not None:
        if allowed and requested not in allowed:
            raise MeshPipelineError(f"Unsupported boundary_layer.grow_on={requested!r}; allowed: {allowed}")
        state["grow_on"] = requested
    elif allowed:
        normalised = {str(value).lower().replace("-", "").replace(" ", ""): value for value in allowed}
        chosen = next((normalised[key] for key in (
            "selectededgelabels", "selectedlabels", "selectededges", "walls", "allwalls"
        ) if key in normalised), None)
        if chosen is None:
            raise MeshPipelineError(
                "Cannot select body-wall boundary layers with the exposed GrowOn "
                f"values {allowed}; specify a documented boundary_layer.grow_on."
            )
        state["grow_on"] = chosen
    # If this release does not expose enum metadata, preserve its default and
    # the explicit edge-label scope. Confirm inflation in the Windows smoke run.
    scope = state.get("grow_on", task.arguments.get_state().get("grow_on", "release default"))
    print(f"[mesh] Boundary-layer edge labels: {[label]}; GrowOn: {scope}")
    return task.add_child_and_update(state=state, defer_update=False)


def build_mesh(cfg: dict, geometry_file: str | Path) -> Path:
    """Build a fresh planar mesh; never silently remove boundary layers."""
    geometry_file = Path(geometry_file).resolve()
    if not geometry_file.is_file() or geometry_file.stat().st_size == 0:
        raise MeshPipelineError(f"Labeled geometry is missing or empty: {geometry_file}")
    try:
        import ansys.fluent.core as pyfluent
    except ImportError as exc:
        raise MeshPipelineError("Install the project's PyFluent dependencies.") from exc
    paths = work_paths(cfg)
    mesh_file = paths["mesh"]
    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    staged = mesh_file.with_name(f".02_mesh_{uuid.uuid4().hex}.msh.h5")
    log_dir = paths["work"] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    runtime = cfg.get("runtime", {})
    mesh = cfg["mesh"]
    body_name = cfg["geometry"].get("body_wall_name", "body_wall")
    farfield_name = cfg["geometry"].get("farfield_name", "farfield")
    session = None
    stage = "launch"
    print("[mesh] Launching 3-D double-precision meshing host for the 2-D workflow...")
    try:
        session = pyfluent.launch_fluent(
            mode=pyfluent.FluentMode.MESHING,
            dimension=pyfluent.Dimension.THREE,
            precision=pyfluent.Precision.DOUBLE,
            product_version=runtime.get("fluent_version", "25.1.0"),
            processor_count=int(runtime.get("processors", 4)),
            ui_mode=pyfluent.UIMode.GUI if runtime.get("fluent_gui", False) else pyfluent.UIMode.NO_GUI,
            cwd=str(log_dir),
            start_timeout=int(runtime.get("fluent_start_timeout_s", 300)),
        )
        session.meshing.GlobalSettings.EnablePrime2dMeshing.set_state(True)
        workflow = session.two_dimensional_meshing(legacy=True)
        stage = "cad_import"
        print(f"[mesh] Loading 2-D CAD: {geometry_file}")
        _run_task(workflow, ("load_cad_geometry_2d", "load_cad_geometry"), {
            "file_name": str(geometry_file), "length_unit": mesh.get("length_unit", "m")
        })
        stage = "regions"
        print("[mesh] Updating fluid regions")
        regions = _workflow_task(workflow, "update_regions_2d", "update_regions")
        if regions() is False:
            raise MeshPipelineError("Fluent reported failure while updating fluid regions.")
        region_state = regions.arguments.get_state()
        region_names = region_state.get("region_current_list") or region_state.get("region_name_list", [])
        region_types = region_state.get("region_current_type_list") or region_state.get("region_type_list", [])
        if not region_names or len(region_names) != len(region_types) or any(
            str(kind).casefold() != "fluid" for kind in region_types
        ):
            raise MeshPipelineError(f"Expected fluid regions after update; received {region_state}")
        print(f"[mesh] Confirmed {len(region_names)} fluid region(s)")
        stage = "boundaries"
        print("[mesh] Updating body and farfield boundaries")
        _run_task(workflow, ("update_boundaries_2d", "update_boundaries"), {
            "selection_type": "label",
            "boundary_label_list": [body_name, farfield_name],
            "boundary_label_type_list": ["wall", "pressure-far-field"],
        })
        stage = "sizing"
        print("[mesh] Defining global and body curvature sizing")
        _run_task(workflow, ("define_global_sizing_2d", "define_global_sizing"), {
            "min_size": float(mesh["global_min_size"]),
            "max_size": float(mesh["global_max_size"]),
            "curvature_normal_angle": float(mesh.get("curvature_normal_angle_deg", 18.0)),
        })
        _make_curvature_sizing(workflow, body_name, cfg)
        stage = "initial_surface_mesh"
        print("[mesh] Generating initial 2-D surface mesh")
        _run_task(workflow, ("generate_initial_surface_mesh",))
        layers = mesh.get("boundary_layer", {})
        if layers.get("enabled", True):
            stage = "boundary_layer"
            print("[mesh] Adding boundary layers")
            _add_boundary_layers(workflow, body_name, layers)
        stage = "write_mesh"
        print(f"[mesh] Writing 2-D mesh: {mesh_file}")
        _run_task(workflow, ("write_2d_mesh",), {
            "file_name": str(staged), "skip_export": False
        })
        if not staged.is_file() or staged.stat().st_size == 0:
            raise MeshPipelineError(f"Fluent did not create a nonempty 2-D mesh: {staged}")
        staged.replace(mesh_file)
        print(f"[mesh] Complete: {mesh_file}")
        return mesh_file
    except Exception as exc:
        raise MeshPipelineError(f"2-D meshing failed during {stage}: {exc}", stage=stage) from exc
    finally:
        if session is not None:
            try:
                session.exit()
            except Exception as exc:
                print(f"[mesh] Warning: owned meshing session could not exit: {exc}")
        if staged.exists():
            try:
                staged.unlink()
            except OSError as exc:
                print(f"[mesh] Warning: temporary mesh could not be removed: {exc}")
