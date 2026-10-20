"""Scientific bar metrics retain peaks, signed energy and unit-span meaning."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import surface_reporting
from pipeline.reporting import read_csv


def sample(time, index, convective, radiative):
    return dict(time_s=time, surface_order_index=index,
                convective_heat_flux_into_solid_W_m2=convective,
                radiative_heat_flux_out_W_m2=radiative,
                net_heat_flux_into_solid_W_m2=convective-radiative)


class HeatLoadBarTests(unittest.TestCase):
    def test_flux_uses_earlier_spatial_peak_and_load_uses_final_global_signed_energy(self):
        cases = {"blunt": dict(
            flux=[sample(0, 4, 900, 100), sample(0, 5, 1000, 300), sample(2, 4, 200, 400)],
            # These deliberately disagree with integrated global energy. Local
            # J/m² must not be substituted or summed as total J/m of span.
            energy=[dict(time_s=2, cumulative_net_energy_J_m2=999999)],
            **{"global": [dict(time_s=2, cumulative_convective_energy_per_unit_span_J_m=1000,
                              cumulative_radiative_energy_per_unit_span_J_m=1200,
                              cumulative_net_energy_per_unit_span_J_m=-200),
                         dict(time_s=1, cumulative_convective_energy_per_unit_span_J_m=800,
                              cumulative_radiative_energy_per_unit_span_J_m=100,
                              cumulative_net_energy_per_unit_span_J_m=700)]}),
            "sharp": dict(flux=[sample(0, 8, 2000, 0), sample(2, 8, 1500, 100)])}
        saved_bars = {}
        original = surface_reporting._save
        def capture(fig, path, plt, dpi=160):
            saved_bars[Path(path).name] = [[bar.get_height() for bar in ax.patches] for ax in fig.axes]
            return original(fig, path, plt, dpi)
        with tempfile.TemporaryDirectory() as directory, patch.object(surface_reporting, "_save", side_effect=capture):
            outputs = surface_reporting.surface_load_bar_outputs(cases, Path(directory), {"blunt": "blue", "sharp": "orange"})
            flux = read_csv(outputs["peak_heat_flux_comparison_csv"])
            self.assertEqual(float(flux[0]["peak_convective_heat_flux_W_m2"]), 1000.)
            self.assertEqual(float(flux[0]["time_of_peak_convective_flux_s"]), 0.)
            self.assertEqual(int(flux[0]["peak_convective_surface_order_index"]), 5)
            self.assertEqual(float(flux[0]["peak_net_heat_flux_W_m2"]), 800.)
            self.assertEqual(float(flux[0]["peak_radiative_heat_flux_W_m2"]), 400.)
            loads = read_csv(outputs["total_heat_load_comparison_csv"])
            self.assertEqual([row["design"] for row in loads], ["blunt"])
            self.assertEqual(float(loads[0]["final_time_s"]), 2.)
            self.assertEqual(float(loads[0]["final_net_heat_load_per_unit_span_J_m"]), -200.)
            self.assertEqual(saved_bars["total_heat_load_comparison.png"][2], [-200.])
            self.assertEqual(saved_bars["peak_heat_flux_comparison.png"][0], [1000., 2000.])
            self.assertTrue(all(Path(path).is_file() for path in outputs.values()))

    def test_invalid_flux_and_duplicate_global_times_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "finite"):
                surface_reporting.surface_load_bar_outputs({"nose": {"flux": [sample(0, 1, float("nan"), 0)]}},
                    Path(directory), {"nose": "blue"})
            with self.assertRaisesRegex(ValueError, "unique"):
                surface_reporting.surface_load_bar_outputs({"nose": {"global": [{"time_s": 1}, {"time_s": 1}]}},
                    Path(directory), {"nose": "blue"})
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":unittest.main()
