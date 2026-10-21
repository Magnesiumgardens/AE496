"""Steady sample maxima and solved transient maxima remain distinct metrics."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure

from pipeline.reporting import _plotting, read_csv
from pipeline.surface_reporting import peak_temperature_outputs


def gas_case(name, wall, fluid, mode="radiative_equilibrium"):
    return dict(cfg=dict(project=dict(name=name), thermal=dict(mode=mode)),
                values=[dict(wall_temperature_K=a, near_wall_fluid_temperature_K=b)
                        for a, b in zip(wall, fluid)])


class TemperatureBarTests(unittest.TestCase):
    def capture_figures(self, saved):
        original = Figure.savefig
        def capture(fig, path, *args, **kwargs):
            saved[Path(path).name] = dict(
                bars=[[bar.get_height() for bar in ax.patches] for ax in fig.axes],
                labels=[ax.get_ylabel() for ax in fig.axes],
                titles=[ax.get_title() for ax in fig.axes]
                       + ([fig._suptitle.get_text()] if fig._suptitle is not None else []))
            return original(fig, path, *args, **kwargs)
        return patch.object(Figure, "savefig", new=capture)

    def test_steady_bars_use_per_nose_sample_maxima_with_kelvin_units(self):
        from pipeline.temperature_comparison import steady_temperature_bars
        cases = [gas_case("blunt", [900., 1700., 1000.], [1200., 1900., 1400.]),
                 gas_case("sharp", [1500., 2300., 1600.], [1600., 2600., 1700.])]
        figures = {}
        with tempfile.TemporaryDirectory() as temp, self.capture_figures(figures):
            output = steady_temperature_bars(cases, Path(temp), _plotting())
            wall = read_csv(output["raw_max_wall_temperature_comparison_csv"])
            fluid = read_csv(output["raw_max_near_wall_fluid_temperature_comparison_csv"])
            self.assertEqual([row["design"] for row in wall], ["blunt", "sharp"])
            self.assertEqual([float(row["maximum_wall_temperature_K"]) for row in wall], [1700., 2300.])
            self.assertEqual([float(row["maximum_near_wall_fluid_temperature_K"]) for row in fluid], [1900., 2600.])
            self.assertEqual(figures["raw_max_wall_temperature_comparison.png"]["bars"], [[1700., 2300.]])
            self.assertEqual(figures["raw_max_near_wall_fluid_temperature_comparison.png"]["bars"], [[1900., 2600.]])
            for name in ("raw_max_wall_temperature_comparison.png", "raw_max_near_wall_fluid_temperature_comparison.png"):
                self.assertTrue(all("[K]" in label for label in figures[name]["labels"]))
            self.assertTrue(all(Path(path).is_file() for path in output.values()))

    def test_steady_nonfinite_temperature_is_rejected(self):
        from pipeline.temperature_comparison import steady_temperature_bars
        for wall, fluid in (([float("nan")], [300.]), ([300.], [float("inf")])):
            with self.subTest(wall=wall, fluid=fluid), tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(ValueError):
                    steady_temperature_bars([gas_case("nose", wall, fluid)], Path(temp), _plotting())

    def test_solid_bars_keep_earlier_peak_separate_from_final_200_second_value(self):
        histories = {
            "blunt": [dict(time_s=0., maximum_K=300., mean_K=300.),
                      dict(time_s=120., maximum_K=1400., mean_K=600.),
                      dict(time_s=200., maximum_K=900., mean_K=500.)],
            "sharp": [dict(time_s=0., maximum_K=300., mean_K=300.),
                      dict(time_s=200., maximum_K=1700., mean_K=700.)],
        }
        figures = {}
        with tempfile.TemporaryDirectory() as temp, self.capture_figures(figures):
            output = peak_temperature_outputs(histories, Path(temp), {"blunt": "blue", "sharp": "orange"})
            peaks = read_csv(output["max_temperature_comparison_csv"])
            final = read_csv(output["final_temperature_comparison_csv"])
            self.assertEqual([float(row["peak_temperature_K"]) for row in peaks], [1400., 1700.])
            self.assertEqual(float(peaks[0]["time_of_peak_s"]), 120.)
            self.assertEqual([float(row["final_temperature_K"]) for row in final], [900., 1700.])
            self.assertEqual([float(row["final_time_s"]) for row in final], [200., 200.])
            self.assertEqual(figures["final_temperature_comparison.png"]["bars"], [[900., 1700.]])
            self.assertIn("200", " ".join(figures["max_temperature_comparison.png"]["titles"]))
            self.assertIn("200", " ".join(figures["final_temperature_comparison.png"]["titles"]))

    def test_400_second_exposure_is_not_cut_off_or_labeled_as_200_seconds(self):
        histories = {"nose": [dict(time_s=0., maximum_K=300.), dict(time_s=200., maximum_K=800.),
                               dict(time_s=350., maximum_K=1800.), dict(time_s=400., maximum_K=1500.)]}
        figures = {}
        with tempfile.TemporaryDirectory() as temp, self.capture_figures(figures):
            output = peak_temperature_outputs(histories, Path(temp), {"nose": "blue"})
            peak = read_csv(output["max_temperature_comparison_csv"])[0]
            final = read_csv(output["final_temperature_comparison_csv"])[0]
            self.assertEqual(float(peak["peak_temperature_K"]), 1800.)
            self.assertEqual(float(peak["time_of_peak_s"]), 350.)
            self.assertEqual(float(final["final_temperature_K"]), 1500.)
            self.assertEqual(float(final["final_time_s"]), 400.)
            for name in ("max_temperature_comparison.png", "final_temperature_comparison.png"):
                title = " ".join(figures[name]["titles"])
                self.assertIn("400", title)
                self.assertNotIn("200", title)

    def test_initial_solid_state_remains_part_of_peak_for_a_cooling_history(self):
        histories = {"nose": [dict(time_s=0., maximum_K=300.), dict(time_s=200., maximum_K=250.)]}
        with tempfile.TemporaryDirectory() as temp:
            output = peak_temperature_outputs(histories, Path(temp), {"nose": "blue"})
            peak = read_csv(output["max_temperature_comparison_csv"])[0]
            final = read_csv(output["final_temperature_comparison_csv"])[0]
            self.assertEqual(float(peak["peak_temperature_K"]), 300.)
            self.assertEqual(float(peak["time_of_peak_s"]), 0.)
            self.assertEqual(float(final["final_temperature_K"]), 250.)


if __name__ == "__main__": unittest.main()
