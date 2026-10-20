"""Independent FE volume integrals and missing-radiation energy detection."""
import unittest

import numpy as np

from pipeline.aerothermal_loading import STEFAN_BOLTZMANN
from pipeline.thermal_energy import evaluate_energy_balance, normalize_solid_connectivity, prepare_solid_energy
from pipeline.thermal_loading import ThermalAnalysisError


def triangle():
    numbers = np.array([11, 31, 7, 80, 5, 103])
    xy = np.array([[0., 0.], [1., 0.], [0., 1.], [.5, 0.], [.5, .5], [0., .5]])
    return numbers, np.c_[xy, np.zeros(6)], numbers[None, :]


class SolidEnergyTests(unittest.TestCase):
    def test_native_degenerate_eight_node_records_are_normalized_geometrically(self):
        nnum, xyz, cells = triangle()
        header = [1, 1, 0, 0, 0, 0, 0, 0, 7, 0]
        # Repeated K and a zero midside slot reproduce a triangular element
        # encoded through an eight-node native solid connectivity.
        native = header + [11, 31, 7, 7, 80, 5, 0, 103]
        normalized = normalize_solid_connectivity(nnum, xyz, [native])
        energy = prepare_solid_energy(nnum, xyz, normalized)
        self.assertAlmostEqual(energy.area_m2, .5)
        x, y = xyz[:, 0], xyz[:, 1]
        expected = 10. / 12. + 6. / 24. + 8. / 12.
        self.assertAlmostEqual(energy.stored_energy_change(300. + 10.*x*x + 6.*x*y + 8.*y*y,
                                                          300., 1., 1.), expected)
        # Even another native ordering is unambiguous when the geometry and
        # midside locations are correct. No raw-node-position guess is used.
        reordered = header + [5, 103, 31, 11, 80, 7]
        np.testing.assert_allclose(prepare_solid_energy(nnum, xyz,
            normalize_solid_connectivity(nnum, xyz, [reordered])).nodal_volume_weights_m2,
            energy.nodal_volume_weights_m2, atol=1e-15)
        bad_type = native.copy(); bad_type[1] = 2
        bad_midpoint = xyz.copy(); bad_midpoint[3, 0] += .01
        for changed_xyz, records in ((xyz, [bad_type]), (bad_midpoint, [native]), (xyz, [])):
            with self.assertRaises(ThermalAnalysisError):
                normalize_solid_connectivity(nnum, changed_xyz, records)

    def test_quadratic_FE_integral_is_not_an_arithmetic_node_mean(self):
        nnum, xyz, cells = triangle()
        quadrature = prepare_solid_energy(nnum, xyz, cells)
        self.assertAlmostEqual(quadrature.area_m2, .5)
        # On the unit triangle, integral(x²)=integral(y²)=1/12,
        # integral(x*y)=1/24. Integrating these independent physical fields
        # tests the complete corner/midside interpolation and area measure.
        x, y = xyz[:, 0], xyz[:, 1]
        delta = 10. * x*x + 6. * x*y + 8. * y*y
        actual = quadrature.stored_energy_change(300. + delta, 300., 2., 3.)
        expected = 2. * 3. * ((10. + 8.) / 12. + 6. / 24.)
        self.assertAlmostEqual(actual, expected, places=12)
        self.assertNotAlmostEqual(actual, .5 * 2. * 3. * np.mean(delta))
        np.testing.assert_allclose(quadrature.nodal_volume_weights_m2[:3], 0., atol=1e-15)
        np.testing.assert_allclose(quadrature.nodal_volume_weights_m2[3:], 1. / 6., atol=1e-15)

    def test_linear_uniform_and_reversed_element_orientation(self):
        nnum, xyz, cells = triangle()
        for connectivity in (cells, cells[:, [0, 2, 1, 5, 4, 3]]):
            quadrature = prepare_solid_energy(nnum, xyz + [8., -3., 0.], connectivity)
            delta = 2. + 3. * xyz[:, 0] + 6. * xyz[:, 1]
            expected = .5 * (2. + 3. / 3. + 6. / 3.)
            self.assertAlmostEqual(quadrature.stored_energy_change(300. + delta, 300., 1., 1.), expected)
            self.assertAlmostEqual(quadrature.stored_energy_change(np.full(6, 400.), 300., 2., 3.), 300.)

    def test_energy_balance_matches_independent_implicit_radiative_lumped_model(self):
        nnum, xyz, cells = triangle()
        quadrature = prepare_solid_energy(nnum, xyz, cells)
        settings = {"initial_t": 300., "rho": 2., "cp": 3.}
        capacity, h_area, driving = .5 * 2. * 3., 5., 800.
        times = np.array([0., .01, .05, .2, .5, 1.])
        temperatures = [300.]
        def power(temperature):
            convective = h_area * (driving - temperature)
            radiative = .45 * STEFAN_BOLTZMANN * (temperature**4 - 3.**4)
            return [convective, radiative, convective - radiative]
        for dt in np.diff(times):
            previous = temperatures[-1]
            low, high = previous, driving
            for _ in range(100):
                candidate = .5 * (low + high)
                residual = capacity * (candidate - previous) - dt * power(candidate)[2]
                if residual > 0: high = candidate
                else: low = candidate
            temperatures.append(.5 * (low + high))
        values = np.repeat(np.asarray(temperatures)[:, None], 6, axis=1)
        rows, summary = evaluate_energy_balance(quadrature, times, values,
                                                [power(value) for value in temperatures], settings)
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["all_steps_passed"])
        self.assertTrue(summary["final_balance_passed"])
        self.assertLess(summary["maximum_absolute_step_error_J_m"], 1e-10)
        self.assertEqual(summary["solved_substeps_checked"], 5)
        self.assertEqual(rows[0]["solid_internal_energy_change_per_unit_span_J_m"], 0.)
        self.assertAlmostEqual(summary["final_stored_energy_change_per_unit_span_J_m"],
                               capacity * (temperatures[-1] - 300.), places=10)
        self.assertNotAlmostEqual(summary["final_trapezoidal_net_energy_per_unit_span_J_m"],
                                  summary["final_implicit_net_energy_per_unit_span_J_m"])

    def test_convection_only_native_solution_cannot_pass_postprocessed_radiation_balance(self):
        nnum, xyz, cells = triangle()
        quadrature = prepare_solid_energy(nnum, xyz, cells)
        settings = {"initial_t": 300., "rho": 2., "cp": 10.}
        capacity, h_area, driving = 10., 10., 3000.
        # Use capacity=area*rho*cp. Each exact backward-Euler step below
        # deliberately excludes radiation, reproducing the failure mechanism.
        self.assertEqual(capacity, .5 * settings["rho"] * settings["cp"])
        times = np.array([0., 1., 2., 4.])
        temperatures = [300.]
        for dt in np.diff(times):
            temperatures.append((capacity * temperatures[-1] + dt * h_area * driving) / (capacity + dt * h_area))
        values = np.repeat(np.asarray(temperatures)[:, None], 6, axis=1)
        convective = h_area * (driving - np.asarray(temperatures))
        radiative = .45 * STEFAN_BOLTZMANN * (np.asarray(temperatures)**4 - 3.**4)
        powers = np.column_stack((convective, radiative, convective - radiative))
        rows, summary = evaluate_energy_balance(quadrature, times, values, powers, settings)
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["all_steps_passed"])
        self.assertFalse(summary["final_balance_passed"])
        self.assertFalse(summary["energy_sign_passed"])
        self.assertGreater(summary["final_stored_energy_change_per_unit_span_J_m"], 0.)
        self.assertLess(summary["final_implicit_net_energy_per_unit_span_J_m"], 0.)
        self.assertFalse(rows[-1]["cumulative_energy_balance_passed"])

    def test_step_gate_does_not_hide_cancelling_earlier_errors(self):
        nnum, xyz, cells = triangle()
        quadrature = prepare_solid_energy(nnum, xyz, cells)
        settings = {"initial_t": 300., "rho": 2., "cp": 1.}
        values = np.array([np.full(6, t) for t in (300., 302., 302.)])
        rows, summary = evaluate_energy_balance(quadrature, [0., 1., 2.], values,
                                                [[1., 0., 1.]] * 3, settings)
        self.assertTrue(summary["final_balance_passed"])
        self.assertFalse(summary["all_steps_passed"])
        self.assertEqual(summary["status"], "failed")

    def test_invalid_mesh_and_incomplete_or_inconsistent_histories_are_rejected(self):
        nnum, xyz, cells = triangle()
        for bad in (cells[:, :3], np.array([[11, 31, 7, 80, 5, 5]]),
                    np.array([[11, 31, 7, 80, 5, 999]]), np.repeat(cells, 2, axis=0)):
            with self.subTest(connectivity=bad):
                with self.assertRaises(ThermalAnalysisError): prepare_solid_energy(nnum, xyz, bad)
        with self.assertRaises(ThermalAnalysisError): prepare_solid_energy(nnum, xyz * 0, cells)
        quadrature = prepare_solid_energy(nnum, xyz, cells)
        settings = {"initial_t": 300., "rho": 2., "cp": 3.}
        values, powers = np.full((2, 6), 300.), np.array([[1., 1., 0.]] * 2)
        for times, changed_values, changed_powers in (
                ([1., 2.], values, powers), ([0., 0.], values, powers),
                ([0., 1.], values + 1., powers),
                ([0., 1.], values, np.array([[1., 1., 1.]] * 2))):
            with self.assertRaises(ThermalAnalysisError):
                evaluate_energy_balance(quadrature, times, changed_values, changed_powers, settings)


if __name__ == "__main__":
    unittest.main()
