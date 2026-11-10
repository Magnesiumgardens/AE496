"""Independent property-table, enthalpy and native assignment checks."""
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np
import yaml

from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.thermal_materials import apply_material, from_material_config


def linear_material(**changes):
    cfg = dict(name="linear cp", density_kg_m3=2., temperature_k=[250., 2000.],
               conductivity_w_mk={"x": [10., 20.], "y": [10., 20.], "z": [4., 8.]},
               specific_heat_j_kgk=[950., 2700.])  # cp(T) = 700 + T
    cfg.update(changes)
    return from_material_config(cfg)


class ThermalMaterialsTests(unittest.TestCase):
    def test_reference_xml_table_is_converted_from_celsius_without_property_scaling(self):
        path = Path(__file__).resolve().parents[1] / "materials/reference_carbon_carbon_2d.yaml"
        law = from_material_config(yaml.safe_load(path.read_text()))
        self.assertEqual(law.density, 1600.)
        np.testing.assert_allclose(law.temperature_K[[0, -1]], [255.556, 2200.])
        np.testing.assert_allclose(law.conductivity(law.temperature_K[[0, -1]], "x"), [6.07, 11.7])
        np.testing.assert_allclose(law.conductivity(law.temperature_K[[0, -1]], "z"), [3.89, 7.49])
        np.testing.assert_allclose(law.specific_heat(law.temperature_K[[0, -1]]), [712., 1840.])
        with self.assertRaises(ThermalAnalysisError):law.specific_heat(2300.)

    def test_integral_and_inverse_are_consistent_with_analytic_linear_cp(self):
        law = linear_material()
        temperatures = np.array([[250., 300., 400.], [900., 1200., 2000.]])
        expected = 700. * (temperatures - 300.) + .5 * (temperatures**2 - 300.**2)
        np.testing.assert_allclose(law.enthalpy_change(temperatures), expected)
        np.testing.assert_allclose(law.temperature_from_enthalpy(expected), temperatures)
        self.assertEqual(law.enthalpy_change(300.), 0.)
        self.assertIsInstance(law.specific_heat(400.), float)

    def test_piecewise_cp_integrates_each_interval_and_inverts_decreasing_slopes(self):
        law = linear_material(temperature_k=[250., 300., 400., 600.],
            conductivity_w_mk=10., specific_heat_j_kgk=[800., 1000., 900., 1200.])
        expected = [-45000., 0., 95000., 305000.]
        np.testing.assert_allclose(law.enthalpy_change([250., 300., 400., 600.]), expected)
        np.testing.assert_allclose(law.temperature_from_enthalpy(expected), [250., 300., 400., 600.])

    def test_endpoint_extension_extends_properties_and_enthalpy_without_clipping_temperature(self):
        law = linear_material(extrapolation="hold_endpoint")
        np.testing.assert_allclose(law.specific_heat([200., 2100.]), [950., 2700.])
        enthalpy = law.enthalpy_change([200., 2100.])
        expected = [law.enthalpy_change(250.) - 50. * 950., law.enthalpy_change(2000.) + 100. * 2700.]
        np.testing.assert_allclose(enthalpy, expected)
        np.testing.assert_allclose(law.temperature_from_enthalpy(enthalpy), [200., 2100.])
        diagnostics = law.range_diagnostics([200., 300., 2100.])
        self.assertEqual(diagnostics["status"], "outside_range")
        self.assertEqual(diagnostics["below_table_count"], 1)
        self.assertEqual(diagnostics["above_table_count"], 1)

    def test_constant_material_uses_exact_legacy_sensible_energy(self):
        law = from_material_config(dict(density_kg_m3=19250., conductivity_w_mk=170., specific_heat_j_kgk=130.))
        np.testing.assert_allclose(law.enthalpy_change([300., 800.]), [0., 65000.])
        np.testing.assert_allclose(law.temperature_from_enthalpy([0., 65000.]), [300., 800.])
        self.assertEqual(law.range_diagnostics([300., 800.])["status"], "constant_properties")

    def test_native_axis_mapping_and_six_entry_chunks_preserve_supplied_values(self):
        temperatures = np.arange(300., 1100., 100.)
        law = linear_material(temperature_k=temperatures, conductivity_w_mk={"x": np.arange(10., 18.), "z": np.arange(4., 12.)},
            specific_heat_j_kgk=np.arange(1000., 1800., 100.))
        native = Mock()
        metadata = apply_material(native, law, axes=("z", "x"))
        self.assertEqual(metadata["global_xy_material_axes"], ["z", "x"])
        self.assertEqual(native.mptemp.call_args_list[1].args, (1, *temperatures[:6]))
        self.assertEqual(native.mptemp.call_args_list[2].args, (7, *temperatures[6:]))
        self.assertEqual(native.mpdata.call_args_list[0].args, ("KXX", 1, 1, 4., 5., 6., 7., 8., 9.))
        self.assertEqual(native.mpdata.call_args_list[2].args, ("KYY", 1, 1, 10., 11., 12., 13., 14., 15.))
        self.assertEqual(native.mpdata.call_args_list[-1].args, ("C", 1, 7, 1600., 1700.))
        native.mp.assert_called_once_with("DENS", 1, 2.)
        with self.assertRaises(ThermalAnalysisError):apply_material(native, law, axes=("x", "y"))

    def test_invalid_table_shapes_ranges_and_missing_axes_fail(self):
        for change in (dict(temperature_k=[300., 250.]), dict(temperature_c=[0., 100.]),
                       dict(specific_heat_j_kgk=[1000.]), dict(density_kg_m3=[2.]),
                       dict(conductivity_w_mk={"radial": 10.}), dict(specific_heat_j_kgk=-1.),
                       dict(extrapolation="clip")):
            with self.subTest(change=change), self.assertRaises(ThermalAnalysisError):linear_material(**change)
        with self.assertRaises(ThermalAnalysisError):linear_material().temperature_from_enthalpy(1e10)
        with self.assertRaises(ThermalAnalysisError):linear_material().conductivity(300., "unknown")


if __name__ == "__main__":unittest.main()
