"""Offline transport contracts for the native radiation boundary.

These tests inspect native model construction and selection invariants. They
do not simulate SURF151 heat transfer or substitute for a licensed MAPDL run.
"""
from types import SimpleNamespace
import unittest

import numpy as np

from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.thermal_radiation import (
    SOLID_ELEMENTS, SOLID_NODES, STEFAN_BOLTZMANN,
    add_radiation_surfaces, select_solid_results,
)


def element_record(element_type, nodes):
    return np.asarray([1 if element_type == 1 else 2, element_type,
                       element_type, 0, 0, 0, 0, 0, 0, 0, *nodes], dtype=int)


class SurfaceMeshAdapter:
    """Small command transport with selections and independent edge topology."""

    def __init__(self, nnum, xyz, lines=(101, 102, 103)):
        self.solid_numbers = np.asarray(nnum, dtype=int).copy()
        self.coordinates = dict(zip(map(int, nnum), np.asarray(xyz).copy()))
        # The fixture is one independent native quadratic triangle, not a
        # topology inferred from the production boundary-quadrature objects.
        ids = list(map(int, nnum))
        self.edges = dict(zip(lines, [(ids[0], ids[3], ids[1]),
                                      (ids[1], ids[4], ids[2]),
                                      (ids[2], ids[5], ids[0])]))
        self.records = {1: element_record(1, ids)}
        self.node_selection = set(ids)
        self.element_selection = {1}
        self.line_selection = set(lines)
        self.components = {}
        self.calls = []
        self.failure_command = None
        self.surface_fault = None
        self.surface_generated_from_nodes = None
        self.post_processing = SimpleNamespace(
            nsets=2, time_values=[2., 200.], nodal_temperature=self.temperatures)

    @property
    def mesh(self):
        ids = np.asarray(sorted(self.node_selection), dtype=int)
        return SimpleNamespace(nnum=ids,
                               nodes=np.asarray([self.coordinates[int(node)] for node in ids]),
                               enum=np.asarray(sorted(self.element_selection), dtype=int),
                               elem=[self.records[index] for index in sorted(self.element_selection)])

    def temperatures(self):
        return np.asarray([500. if node in self.solid_numbers else 3.
                           for node in sorted(self.node_selection)])

    def __getattr__(self, name):
        def command(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == self.failure_command:
                return "*** WARNING *** native option was ignored"
            if name == "allsel":
                self.node_selection = set(self.coordinates)
                self.element_selection = set(self.records)
                self.line_selection = set(self.edges)
            elif name == "cm":
                selection = self.node_selection if args[1] == "NODE" else self.element_selection
                self.components[args[0]] = (args[1], selection.copy())
            elif name == "cmsel":
                kind, selection = self.components[args[1]]
                if kind == "NODE": self.node_selection = selection.copy()
                else: self.element_selection = selection.copy()
            elif name == "n":
                self.coordinates[int(args[0])] = np.asarray(args[1:4], dtype=float)
                self.node_selection.add(int(args[0]))
            elif name == "esel":
                self.element_selection = {index for index, record in self.records.items()
                                          if record[1] == int(args[3])}
            elif name == "lsel":
                if args[0] == "NONE": self.line_selection = set()
                elif args[0] == "A": self.line_selection.add(int(args[3]))
            elif name == "nsll":
                self.node_selection = {node for line in self.line_selection for node in self.edges[line]}
            elif name == "esurf":
                ambient = int(args[0])
                self.surface_generated_from_nodes = self.node_selection.copy()
                faces = [edge for edge in self.edges.values() if set(edge) <= self.node_selection]
                if self.surface_fault == "rear": faces = list(self.edges.values())
                if self.surface_fault == "missing": faces = faces[:-1]
                if self.surface_fault == "duplicate": faces = faces + faces[:1]
                for edge in faces:
                    nodes = [edge[0], edge[2], edge[1], ambient]
                    if self.surface_fault == "midside": nodes[2] = ambient
                    if self.surface_fault == "ambient": nodes[-1] = 999
                    self.records[max(self.records) + 1] = element_record(2, nodes)
            elif name in {"k", "l"}:
                return sum(call[0] == name for call in self.calls)
            elif name == "al": return 1
            return ""
        return command


def fixture():
    numbers = np.asarray([11, 23, 47, 51, 72, 89])
    xyz = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                      [.5, 0., 0.], [.5, .5, 0.], [0., .5, 0.]])
    # Last edge is the rear cut: the solid still contains all its nodes.
    load = SimpleNamespace(segment_active=np.asarray([True, True, False]),
                           boundary_quadrature=[(np.asarray([0, 3, 1]),),
                                                (np.asarray([1, 4, 2]),)])
    engine = SurfaceMeshAdapter(numbers, xyz)
    return engine, numbers, xyz, load


class RadiationSurfaceTests(unittest.TestCase):
    def test_exposed_quadratic_edges_radiate_without_constraining_or_adding_solid_capacity(self):
        engine, numbers, xyz, load = fixture()
        result = add_radiation_surfaces(engine, [101, 102, 103], load, numbers, xyz,
                                       dict(emissivity=.45, trad=3.))
        ambient = result["ambient_node_id"]
        self.assertNotIn(ambient, numbers)
        self.assertNotIn(ambient, engine.surface_generated_from_nodes)
        self.assertEqual(result["surface_elements"], 2)
        actual = [set(record[10:]) for record in engine.records.values() if record[1] == 2]
        self.assertCountEqual(actual, [{11, 51, 23, ambient}, {23, 72, 47, ambient}])
        self.assertFalse(any(89 in face for face in actual))
        self.assertEqual([args for name, args, _ in engine.calls if name == "d"],
                         [(ambient, "TEMP", 3.)])
        self.assertEqual([args for name, args, _ in engine.calls if name == "mp"],
                         [("EMIS", 2, .45)])
        self.assertIn(("r", (2, 1., STEFAN_BOLTZMANN), {}), engine.calls)
        self.assertFalse(any(name in {"sfl", "sfe", "sf"} for name, _, _ in engine.calls))
        self.assertEqual(engine.components[SOLID_NODES][1], set(numbers))
        self.assertEqual(engine.components[SOLID_ELEMENTS][1], {1})

    def test_rejects_missing_duplicate_rear_or_incomplete_surface_elements(self):
        for fault in ("missing", "duplicate", "rear", "midside", "ambient"):
            with self.subTest(fault=fault):
                engine, numbers, xyz, load = fixture()
                engine.surface_fault = fault
                with self.assertRaises(ThermalAnalysisError):
                    add_radiation_surfaces(engine, [101, 102, 103], load, numbers, xyz,
                                           dict(emissivity=.45, trad=3.))

    def test_native_warnings_do_not_produce_a_configured_radiation_boundary(self):
        for command in ("et", "keyopt", "r", "mp", "esurf"):
            with self.subTest(command=command):
                engine, numbers, xyz, load = fixture()
                engine.failure_command = command
                with self.assertRaisesRegex(ThermalAnalysisError, "not accepted"):
                    add_radiation_surfaces(engine, [101, 102, 103], load, numbers, xyz,
                                           dict(emissivity=.45, trad=3.))

    def test_rejects_absent_verified_boundary_topology(self):
        engine, numbers, xyz, load = fixture()
        load.boundary_quadrature = []
        with self.assertRaisesRegex(ThermalAnalysisError, "verified quadratic"):
            add_radiation_surfaces(engine, [101, 102, 103], load, numbers, xyz,
                                   dict(emissivity=.45, trad=3.))
        self.assertFalse(engine.calls)

    def test_solid_result_selection_excludes_ambient_and_surface_elements(self):
        engine, numbers, xyz, load = fixture()
        add_radiation_surfaces(engine, [101, 102, 103], load, numbers, xyz,
                               dict(emissivity=.45, trad=3.))
        self.assertEqual(engine.mesh.nnum.size, numbers.size + 1)
        select_solid_results(engine, numbers)
        np.testing.assert_array_equal(engine.mesh.nnum, numbers)
        self.assertEqual(engine.element_selection, {1})
        np.testing.assert_array_equal(engine.temperatures(), np.full(len(numbers), 500.))
        engine.components[SOLID_NODES][1].add(90)
        with self.assertRaisesRegex(ThermalAnalysisError, "selection changed"):
            select_solid_results(engine, numbers)


if __name__ == "__main__": unittest.main()
