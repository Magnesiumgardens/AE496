"""Physical enthalpy checks independent of endpoint/nodal heat-capacity shortcuts."""
import math
import unittest

import numpy as np

from pipeline.thermal_energy import evaluate_energy_balance, prepare_solid_energy
from pipeline.thermal_loading import ThermalAnalysisError
from test_thermal_energy import triangle
from test_thermal_materials import linear_material


def monomial_integral(px, py):
    # Exact x^px*y^py integral over x>=0,y>=0,x+y<=1.
    return math.factorial(px) * math.factorial(py) / math.factorial(px + py + 2)


class VariableCapacityEnergyTests(unittest.TestCase):
    def test_enthalpy_is_integrated_at_interpolated_temperature_not_interpolated_from_nodes(self):
        numbers, xyz, cells = triangle()
        quadrature = prepare_solid_energy(numbers, xyz, cells)
        law = linear_material()
        polynomial = {(2, 0): 10., (1, 1): 6., (0, 2): 8.}
        delta = sum(coef * xyz[:, 0]**px * xyz[:, 1]**py for (px, py), coef in polynomial.items())
        # cp=1000+dT gives h=1000*dT+dT²/2; integrate each monomial analytically.
        first = sum(coef * monomial_integral(px, py) for (px, py), coef in polynomial.items())
        second = sum(ca * cb * monomial_integral(ax+bx, ay+by)
            for (ax, ay), ca in polynomial.items() for (bx, by), cb in polynomial.items())
        expected = law.density * (1000. * first + .5 * second)
        actual = quadrature.stored_energy_change(300. + delta, 300., law.density, 1000., material_law=law)
        self.assertAlmostEqual(actual, expected, places=10)
        nodal_shortcut = law.density * (law.enthalpy_change(300. + delta) @ quadrature.nodal_volume_weights_m2)
        self.assertGreater(abs(actual - nodal_shortcut), .5)
        np.testing.assert_allclose(quadrature.stored_energy_change(np.array([np.full(6, 300.), 300.+delta]),
            300., law.density, 1000., material_law=law), [0., expected], atol=1e-10)

    def test_exact_uniform_enthalpy_balance_passes_and_constant_cp_shortcut_does_not(self):
        numbers, xyz, cells = triangle()
        quadrature = prepare_solid_energy(numbers, xyz, cells)
        law = linear_material()
        temperatures = np.repeat(np.array([300., 400., 500.])[:, None], 6, axis=1)
        # rho*area=1 kg/m. Independent cp integral gives 105 and 115 kJ/m per step.
        powers = np.array([[100000., 0., 100000.], [105000., 0., 105000.], [115000., 0., 115000.]])
        settings = dict(initial_t=300., rho=2., cp=1000.)
        rows, summary = evaluate_energy_balance(quadrature, [0., 1., 2.], temperatures, powers, settings, material_law=law)
        self.assertEqual(summary["status"], "passed")
        self.assertAlmostEqual(rows[-1]["solid_internal_energy_change_per_unit_span_J_m"], 220000.)
        _, shortcut = evaluate_energy_balance(quadrature, [0., 1., 2.], temperatures, powers, settings)
        self.assertEqual(shortcut["status"], "failed")

    def test_disagreeing_material_density_is_rejected(self):
        numbers, xyz, cells = triangle()
        quadrature = prepare_solid_energy(numbers, xyz, cells)
        with self.assertRaises(ThermalAnalysisError):
            quadrature.stored_energy_change(np.full(6, 400.), 300., 3., 1000., material_law=linear_material())


if __name__ == "__main__":unittest.main()
