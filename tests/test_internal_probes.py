"""Physical depth definitions and independent quadratic-field probe checks."""
import unittest

import numpy as np

from pipeline.internal_probes import axial_tip_probe, prepare_probe_interpolation, surface_normal_probes
from pipeline.thermal_loading import ThermalAnalysisError


class InternalProbeTests(unittest.TestCase):
    def test_axial_ten_mm_probe_is_not_ten_mm_from_every_heated_surface(self):
        polygon = np.asarray([[0., .02], [.1, 0.], [0., -.02]])
        np.testing.assert_allclose(axial_tip_probe(polygon, .01), [.09, 0.], atol=1e-14)
        rotated = np.column_stack((-polygon[:, 1], polygon[:, 0]))
        np.testing.assert_allclose(axial_tip_probe(rotated, .01, [0., 1.]), [0., .09], atol=1e-14)
        with self.assertRaisesRegex(ThermalAnalysisError, "outside"):
            axial_tip_probe(polygon, .11)

    def test_flat_tip_uses_geometric_midpoint_independent_of_extra_sampling(self):
        polygon = np.asarray([[0., 0.], [.1, 0.], [.1, .002], [.1, .004], [.1, .02], [0., .02]])
        np.testing.assert_allclose(axial_tip_probe(polygon, .01), [.09, .01], atol=1e-14)

    def test_normal_depth_is_exact_and_independent_of_polygon_orientation(self):
        polygon = np.asarray([[0., 0.], [.04, 0.], [.04, .04], [0., .04]])
        expected = np.asarray([[.02, .01], [.03, .02], [.02, .03], [.01, .02]])
        for xy in (polygon, polygon[::-1]):
            result = surface_normal_probes(xy, [1, 1, 1, 1], .01, require_min_clearance=True)
            actual = result["points"]
            self.assertEqual(result["status"], "ready")
            self.assertEqual(len(actual), 4)
            for point in expected:self.assertTrue(np.any(np.linalg.norm(actual - point, axis=1) < 1e-13))
            for row in result["probes"]:
                source = np.asarray([row["source_x_m"], row["source_y_m"]])
                point = np.asarray([row["x_m"], row["y_m"]])
                self.assertAlmostEqual(np.linalg.norm(point - source), .01)
                self.assertAlmostEqual(row["nearest_heated_surface_distance_m"], .01)

    def test_normal_offset_can_be_closer_to_opposite_hot_surface_and_is_reported(self):
        polygon = np.asarray([[0., 0.], [.04, 0.], [.04, .015], [0., .015]])
        loose = surface_normal_probes(polygon, [1, 1, 1, 1], .01)
        self.assertEqual(len(loose["probes"]), 4)
        bottom = next(row for row in loose["probes"] if row["source_segment"] == 0)
        self.assertAlmostEqual(bottom["nearest_heated_surface_distance_m"], .005)
        self.assertFalse(bottom["minimum_heated_surface_clearance_satisfied"])
        strict = surface_normal_probes(polygon, [1, 1, 1, 1], .01, require_min_clearance=True)
        self.assertEqual(strict["status"], "unavailable")
        self.assertEqual(strict["points"].shape, (0, 2))
        self.assertEqual(len(strict["excluded"]), 4)

    def test_too_thin_offsets_are_excluded_and_inactive_rear_does_not_receive_probe(self):
        polygon = np.asarray([[0., 0.], [.04, 0.], [.04, .005], [0., .005]])
        result = surface_normal_probes(polygon, [1, 1, 1, 0], .01)
        self.assertEqual([row["source_segment"] for row in result["probes"]], [1])
        self.assertEqual({row["source_segment"] for row in result["excluded"]}, {0, 2})
        self.assertTrue(all("outside" in row["reason"] for row in result["excluded"]))

    def test_six_node_interpolation_reproduces_quadratic_field_and_time_history(self):
        numbers = np.asarray([11, 31, 7, 80, 5, 103])
        xy = np.asarray([[0., 0.], [1., 0.], [0., 1.], [.5, 0.], [.5, .5], [0., .5]])
        xyz = np.c_[xy, np.zeros(6)]
        points = np.asarray([[.2, .3], [.1, .1], [.5, .5], [0., .25]])
        function = lambda x, y: 300. + 40*x + 15*y + 10*x*x + 22*x*y + 7*y*y
        sample = prepare_probe_interpolation(numbers, xyz, numbers[None], points)
        nodal = function(xy[:, 0], xy[:, 1])
        expected = function(points[:, 0], points[:, 1])
        np.testing.assert_allclose(sample.evaluate(nodal), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(sample.evaluate(np.asarray([nodal, nodal + 100.])),
                                   np.asarray([expected, expected + 100.]), rtol=0, atol=1e-12)
        self.assertGreater(np.min(np.abs(nodal - expected[0])), .1)
        with self.assertRaisesRegex(ThermalAnalysisError, "outside"):
            prepare_probe_interpolation(numbers, xyz, numbers[None], [[.8, .8]])

    def test_rotated_shifted_mesh_and_shared_element_edge_use_actual_shape_functions(self):
        xy = np.asarray([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [.5, 0.],
                         [1., .5], [.5, .5], [.5, 1.], [0., .5]])
        numbers = np.arange(10, 19)
        triangles = numbers[np.asarray([[0, 1, 2, 4, 5, 6], [0, 2, 3, 6, 7, 8]])]
        rotation = np.asarray([[0., -1.], [1., 0.]])
        offset = np.asarray([5., -3.])
        transformed = xy @ rotation.T + offset
        points = np.asarray([[.2, .7], [.7, .2], [.5, .5]])
        result = prepare_probe_interpolation(numbers, np.c_[transformed, np.zeros(len(xy))], triangles,
                                             points @ rotation.T + offset)
        field = 300. + 100*xy[:, 0]**2 + 70*xy[:, 1]**2
        np.testing.assert_allclose(result.evaluate(field), 300. + 100*points[:, 0]**2 + 70*points[:, 1]**2,
                                   atol=1e-11)

    def test_bad_geometry_mesh_and_temperature_inputs_are_rejected(self):
        with self.assertRaises(ThermalAnalysisError):axial_tip_probe([[0, 0], [1, 0], [2, 0]], .01)
        with self.assertRaises(ThermalAnalysisError):axial_tip_probe([[0, 0], [1, 0], [0, 1]], 0)
        with self.assertRaises(ThermalAnalysisError):surface_normal_probes([[0, 0], [1, 0], [0, 1]], [0, 0, 0], .01)
        numbers = np.arange(1, 7)
        xy = np.asarray([[0., 0.], [1., 0.], [0., 1.], [.5, 0.], [.5, .5], [0., .5]])
        with self.assertRaisesRegex(ThermalAnalysisError, "straight"):
            changed = xy.copy(); changed[3, 1] += .01
            prepare_probe_interpolation(numbers, changed, numbers[None], [[.2, .2]])
        result = prepare_probe_interpolation(numbers, xy, numbers[None], [[.2, .2]])
        for values in (np.full(5, 300.), np.full(6, -1.), np.full(6, np.nan)):
            with self.assertRaises(ThermalAnalysisError):result.evaluate(values)


if __name__ == "__main__":unittest.main()
