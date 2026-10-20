"""Load plotting preserves physical samples, units, signs and transient peaks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.reporting import (_surface_load_history_outputs, read_csv, _save)
from pipeline.surface_reporting import peak_temperature_outputs


class SurfaceReportingTests(unittest.TestCase):
    def test_profiles_keep_signed_saved_energy_and_use_configured_nose_direction(self):
        # Nose points toward -X, and the rear closure includes a middle vertex
        # that belongs in maps/global histories, but not back->nose->back plots.
        xy = ((0., .1), (-1., 0.), (0., -.1), (0., 0.))
        flux, energy = [], []
        for time in (0., 2.):
            for index, (x, y) in enumerate(xy):
                base = dict(time_s=time, surface_order_index=index, x_m=x, y_m=y)
                q = 100. + index if time == 0 else -10. - index
                flux.append(dict(base, convective_heat_flux_into_solid_W_m2=q + 50,
                                 radiative_heat_flux_out_W_m2=50., net_heat_flux_into_solid_W_m2=q))
                # Use a deliberately different saved integral from q_final*t.
                value = (42. + index) * time
                energy.append(dict(base, cumulative_convective_energy_J_m2=value + 100 * time,
                                   cumulative_radiative_energy_J_m2=100*time, cumulative_net_energy_J_m2=value))
        case = dict(nose_direction=[-1., 0.], flux=flux, energy=energy)
        figures = {}
        from pipeline import surface_reporting
        def capture(fig, path, plt, dpi=160):
            figures[Path(path).name] = [
                dict(title=ax.get_title(), ylabel=ax.get_ylabel(), lines=[np.asarray(line.get_ydata()).copy() for line in ax.lines])
                for ax in fig.axes]
            return _save(fig, path, plt, dpi)
        with tempfile.TemporaryDirectory() as directory, patch.object(surface_reporting, "_save", side_effect=capture):
            created = _surface_load_history_outputs({"nose": case}, Path(directory))
            profile = read_csv(created["final_surface_heat_load_profile_csv"])
            self.assertEqual([int(row["surface_order_index"]) for row in profile], [2, 1, 0])
            np.testing.assert_allclose([float(row["s_norm"]) for row in profile], [0., .5, 1.])
            np.testing.assert_allclose([float(row["cumulative_net_energy_J_m2"]) for row in profile], [88., 86., 84.])
            flux_profile = read_csv(created["final_surface_heat_flux_profile_csv"])
            self.assertEqual([float(row["net_heat_flux_into_solid_W_m2"]) for row in flux_profile], [-12., -11., -10.])
            net_history = figures["surface_heat_flux_max_mean_history.png"][2]
            np.testing.assert_allclose(net_history["lines"][0], [103., -10.])
            np.testing.assert_allclose(net_history["lines"][1], [101.5, -11.5])
            self.assertIn("W/m²", net_history["ylabel"])
            self.assertIn("J/m²", figures["surface_heat_load_max_mean_history.png"][2]["ylabel"])

    def test_bar_uses_earlier_all_solid_peak_not_final_or_wall_temperature(self):
        histories = {
            "blunt": [dict(time_s=0, maximum_K=300), dict(time_s=50, maximum_K=1400), dict(time_s=200, maximum_K=900)],
            "sharp": [dict(time_s=0, maximum_K=300), dict(time_s=200, maximum_K=1700)],
        }
        with tempfile.TemporaryDirectory() as directory:
            outputs = peak_temperature_outputs(histories, Path(directory), {"blunt": "blue", "sharp": "orange"})
            rows = read_csv(outputs["max_temperature_comparison_csv"])
            self.assertEqual([float(row["peak_temperature_K"]) for row in rows], [1400., 1700.])
            self.assertEqual(float(rows[0]["time_of_peak_s"]), 50.)
            self.assertEqual(float(rows[0]["final_time_s"]), 200.)
            self.assertTrue(Path(outputs["max_temperature_comparison_png"]).is_file())
            self.assertIn("all saved solid nodes", rows[0]["sampling"])

    def test_invalid_or_duplicate_time_temperature_data_do_not_make_bars(self):
        with tempfile.TemporaryDirectory() as directory:
            for rows in ([dict(time_s=0, maximum_K=float("nan"))],
                         [dict(time_s=0, maximum_K=300), dict(time_s=0, maximum_K=500)]):
                with self.assertRaises(ValueError):
                    peak_temperature_outputs({"nose": rows}, Path(directory), {"nose": "blue"})
            self.assertFalse((Path(directory) / "max_temperature_comparison.png").exists())


if __name__ == "__main__":unittest.main()
