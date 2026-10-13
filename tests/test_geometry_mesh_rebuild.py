"""CAD topology and meshing failure contracts without a licensed Ansys server."""
import sys
import tempfile
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline.geometry import (
    GeometryPipelineError, _boundary_edges, _label_boundaries, prepare_geometry,
)
from pipeline.mesh import MeshPipelineError, build_mesh


class Surface(Enum):
    SURFACETYPE_PLANE = 1
    SURFACETYPE_CYLINDER = 2


class Loop(Enum):
    INNER_LOOP = "INNER"
    OUTER_LOOP = "OUTER"


class Millimetres:
    def __init__(self, value):
        self.value = value

    def to(self, unit):
        if unit != "m":
            raise ValueError(unit)
        return SimpleNamespace(magnitude=self.value / 1000)


def cad_design(z_mm=0.0):
    inner_edge = SimpleNamespace(id="vehicle")
    outer_edge = SimpleNamespace(id="domain")
    bbox = SimpleNamespace(z=Millimetres(z_mm))
    inner = SimpleNamespace(type=Loop.INNER_LOOP, min_bbox=bbox, max_bbox=bbox, edges=[inner_edge])
    outer = SimpleNamespace(type=Loop.OUTER_LOOP, min_bbox=bbox, max_bbox=bbox, edges=[outer_edge])
    face = SimpleNamespace(surface_type=Surface.SURFACETYPE_PLANE, loops=[inner, outer])
    body = SimpleNamespace(is_surface=True, faces=[face])
    child = SimpleNamespace(bodies=[body], components=[])
    design = SimpleNamespace(bodies=[], components=[child], named_selections=[])
    return design, face


def module_chain(leaf_name, **members):
    parts = leaf_name.split(".")
    modules = {}
    for count in range(1, len(parts) + 1):
        name = ".".join(parts[:count])
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        if count > 1:
            setattr(modules[".".join(parts[:count - 1])], parts[count - 1], module)
    for key, value in members.items():
        setattr(modules[leaf_name], key, value)
    return modules


class GeometryTests(unittest.TestCase):
    def test_nested_body_uses_topological_loop_type_independent_of_order(self):
        design, face = cad_design()
        farfield, body = _boundary_edges(design)
        self.assertEqual([e.id for e in farfield], ["domain"])
        self.assertEqual([e.id for e in body], ["vehicle"])
        face.loops.reverse()
        self.assertEqual(_boundary_edges(design), (farfield, body))

    def test_xy_check_converts_coordinate_quantities_to_metres(self):
        design, _ = cad_design(z_mm=5.0e-6)  # 5 nm, below 10 nm tolerance
        _boundary_edges(design)
        design, _ = cad_design(z_mm=5.0e-4)  # 0.5 micrometre, above tolerance
        with self.assertRaisesRegex(GeometryPipelineError, "z=0"):
            _boundary_edges(design)

    def test_extra_hole_and_nonplanar_surface_are_rejected(self):
        design, face = cad_design()
        face.loops.append(face.loops[0])
        with self.assertRaisesRegex(GeometryPipelineError, "one inner"):
            _boundary_edges(design)
        design, face = cad_design()
        face.surface_type = Surface.SURFACETYPE_CYLINDER
        with self.assertRaisesRegex(GeometryPipelineError, "planar"):
            _boundary_edges(design)

    def test_manual_labels_cannot_swap_body_and_farfield(self):
        design, _ = cad_design()
        farfield, body = _boundary_edges(design)
        design.named_selections = [
            SimpleNamespace(name="body_wall", edges=farfield),
            SimpleNamespace(name="farfield", edges=body),
        ]
        with self.assertRaisesRegex(GeometryPipelineError, "match its CAD loop"):
            _label_boundaries(design, farfield, body, {"auto_label": False})

    def test_export_returns_actual_pmdb_filename_and_closes_owned_modeler(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "input.x_t"
            source.write_text("CAD input placeholder")
            cfg = {"project": {"work_dir": str(Path(td) / "run")},
                   "geometry": {"input_file": str(source)}}
            design, _ = cad_design()
            design.create_named_selection = lambda **kwargs: None
            def export(location):
                result = Path(location) / "imported_design_name.pmdb"
                result.write_bytes(b"PMDB output")
                return result
            design.export_to_pmdb = export
            events = []
            modeler = SimpleNamespace(open_file=lambda path: design, close=lambda: events.append("closed"))
            def launch(**kwargs):
                events.append(kwargs)
                return modeler
            with patch.dict(sys.modules, module_chain("ansys.geometry.core", launch_modeler=launch)):
                result = prepare_geometry(cfg)
            self.assertEqual(result.name, "imported_design_name.pmdb")
            self.assertEqual(result.parent.name, "01_labeled_geometry.pmdb")
            self.assertEqual(events[0], {"mode": "spaceclaim", "version": 251, "timeout": 300})
            self.assertEqual(events[-1], "closed")


class Arguments:
    def __init__(self, state=None):
        self.state = dict(state or {})
        self.grow_on = SimpleNamespace(allowed_values=lambda: ["all-zones", "walls"])

    def update_dict(self, state):
        self.state.update(state)

    def get_state(self):
        return dict(self.state)


class Task:
    def __init__(self, callback=None, state=None):
        self.arguments = Arguments(state)
        self.callback = callback
        self.child_states = []

    def __call__(self):
        if self.callback:
            self.callback(self.arguments.state)
        return True

    def add_child_and_update(self, state=None, defer_update=None):
        self.child_states.append(dict(state))
        self.arguments.update_dict(state)
        self()
        return self


def meshing_session(*, fail_layers=False, write_output=True):
    def layer_callback(state):
        if fail_layers:
            raise RuntimeError("Prism collision")
    def write_callback(state):
        if write_output:
            Path(state["file_name"]).write_bytes(b"mesh data")
    workflow = SimpleNamespace(
        load_cad_geometry_2d=Task(),
        update_regions_2d=Task(state={"region_name_list": [], "region_type_list": [],
                                    "region_current_list": ["region-uuid"], "region_current_type_list": ["fluid"]}),
        update_boundaries_2d=Task(), define_global_sizing_2d=Task(),
        add_local_sizing_2d=Task(), generate_initial_surface_mesh=Task(),
        add_2d_boundary_layers=Task(layer_callback), write_2d_mesh=Task(write_callback),
    )
    closed = []
    enabled = {"value": False}
    def enable(value):
        enabled["value"] = value
    def initialize(legacy):
        if not legacy or not enabled["value"]:
            raise RuntimeError("2-D workflow was not explicitly activated")
        return workflow
    session = SimpleNamespace(two_dimensional_meshing=initialize,
        meshing=SimpleNamespace(GlobalSettings=SimpleNamespace(
            EnablePrime2dMeshing=SimpleNamespace(set_state=enable))),
        exit=lambda: closed.append(True))
    return session, workflow, closed


class MeshingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.pmdb = self.work / "geometry.pmdb"
        self.pmdb.write_bytes(b"PMDB")
        self.cfg = {
            "project": {"work_dir": str(self.work)}, "geometry": {},
            "mesh": {"global_min_size": 0.0002, "global_max_size": 0.02,
                     "body_min_size": 0.00015, "body_max_size": 0.001,
                     "boundary_layer": {"enabled": True}},
        }

    def fluent_modules(self, launch):
        return module_chain("ansys.fluent.core", launch_fluent=launch,
            FluentMode=SimpleNamespace(MESHING="meshing"), Dimension=SimpleNamespace(TWO=2, THREE=3),
            Precision=SimpleNamespace(DOUBLE="double"),
            UIMode=SimpleNamespace(GUI="gui", NO_GUI="no_gui"))

    def test_known_2d_aliases_build_mesh_scoped_to_body_and_close(self):
        session, workflow, closed = meshing_session()
        launch_args = []
        def launch(**kwargs):
            launch_args.append(kwargs)
            return session
        with patch.dict(sys.modules, self.fluent_modules(launch)):
            result = build_mesh(self.cfg, self.pmdb)
        self.assertEqual(result.read_bytes(), b"mesh data")
        self.assertEqual(launch_args[0]["dimension"], 3)
        self.assertEqual(launch_args[0]["start_timeout"], 300)
        self.assertEqual(workflow.add_local_sizing_2d.child_states[0]["edge_label_list"], ["body_wall"])
        layers = workflow.add_2d_boundary_layers.child_states[0]
        self.assertEqual(layers["edge_label_list"], ["body_wall"])
        self.assertEqual(layers["grow_on"], "walls")
        self.assertEqual(closed, [True])

    def test_inflation_error_has_specific_stage_and_preserves_previous_mesh(self):
        mesh = self.work / "02_mesh.msh.h5"
        mesh.write_bytes(b"previous valid mesh")
        session, workflow, closed = meshing_session(fail_layers=True)
        with patch.dict(sys.modules, self.fluent_modules(lambda **kwargs: session)):
            with self.assertRaises(MeshPipelineError) as caught:
                build_mesh(self.cfg, self.pmdb)
        self.assertEqual(caught.exception.stage, "boundary_layer")
        self.assertIn("Prism collision", str(caught.exception))
        self.assertEqual(mesh.read_bytes(), b"previous valid mesh")
        self.assertEqual(closed, [True])
        self.assertTrue(self.cfg["mesh"]["boundary_layer"]["enabled"])

    def test_launch_failure_is_not_a_boundary_layer_failure(self):
        def launch(**kwargs):
            raise TimeoutError("Fluent did not start")
        with patch.dict(sys.modules, self.fluent_modules(launch)):
            with self.assertRaises(MeshPipelineError) as caught:
                build_mesh(self.cfg, self.pmdb)
        self.assertEqual(caught.exception.stage, "launch")

    def test_writer_cannot_report_success_using_a_stale_final_mesh(self):
        mesh = self.work / "02_mesh.msh.h5"
        mesh.write_bytes(b"stale mesh")
        session, _, closed = meshing_session(write_output=False)
        with patch.dict(sys.modules, self.fluent_modules(lambda **kwargs: session)):
            with self.assertRaisesRegex(MeshPipelineError, "nonempty") as caught:
                build_mesh(self.cfg, self.pmdb)
        self.assertEqual(caught.exception.stage, "write_mesh")
        self.assertEqual(mesh.read_bytes(), b"stale mesh")
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
