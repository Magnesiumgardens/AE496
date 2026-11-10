"""Threshold crossings retain brackets and unreached limits remain censored."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.internal_screening_metrics import first_threshold_crossing, screening_bar_outputs
from pipeline.reporting import read_csv


class InternalScreeningTests(unittest.TestCase):
    def test_first_crossing_uses_actual_saved_bracket_even_if_later_cooling_occurs(self):
        result = first_threshold_crossing([0, 100, 150, 200], [300, 1500, 2000, 1600])
        self.assertEqual(result["status"], "reached")
        self.assertEqual(result["time_to_limit_s"], 130.)
        self.assertEqual((result["lower_time_s"], result["upper_time_s"]), (100., 150.))
        self.assertEqual((result["lower_temperature_K"], result["upper_temperature_K"]), (1500., 2000.))
        self.assertEqual(result["peak_temperature_K"], 2000.)
        self.assertEqual(result["time_of_peak_s"], 150.)
        self.assertEqual(result["final_temperature_K"], 1600.)

    def test_exact_sample_initial_threshold_and_unreached_limit_are_distinct(self):
        exact = first_threshold_crossing([0, 143, 200], [300, 1800, 1900])
        self.assertEqual(exact["time_to_limit_s"], 143.)
        initial = first_threshold_crossing([0, 200], [1800, 1700])
        self.assertEqual(initial["status"], "initially_at_or_above_limit")
        self.assertEqual(initial["time_to_limit_s"], 0.)
        cold = first_threshold_crossing([0, 100, 200], [300, 1500, 1700])
        self.assertEqual(cold["status"], "not_reached_within_duration")
        self.assertIsNone(cold["time_to_limit_s"])
        self.assertIsNone(cold["lower_time_s"])
        self.assertEqual(cold["duration_s"], 200.)

    def test_equal_peak_uses_earliest_saved_peak_time(self):
        result = first_threshold_crossing([0, 50, 100], [300, 1500, 1500])
        self.assertEqual(result["time_of_peak_s"], 50.)

    def test_bad_histories_are_not_sorted_clamped_or_filled(self):
        for times, values, limit in (
                ([1, 2], [300, 500], 1800), ([0, 0], [300, 500], 1800),
                ([0, 2, 1], [300, 500, 700], 1800), ([0], [300], 1800),
                ([0, 1], [300, float("nan")], 1800), ([0, 1], [300, 0], 1800),
                ([0, 1], [300], 1800), ([0, float("inf")], [300, 500], 1800),
                ([0, 1], [300, 500], 0)):
            with self.subTest(times=times, values=values, limit=limit), self.assertRaises(ValueError):
                first_threshold_crossing(times, values, limit)

    def test_time_bars_show_lower_bound_and_keep_missing_crossing_in_csv(self):
        reached = first_threshold_crossing([0, 100, 150, 200], [300, 1500, 2000, 1600])
        unreached = first_threshold_crossing([0, 200], [300, 1700])
        rows = [dict(reached, design="sharp", depth_m=.01), dict(unreached, design="blunt", depth_m=.01)]
        from pipeline import reporting
        original = reporting._save
        figures = {}
        def capture(fig, path, plt, **kwargs):
            axis = fig.axes[0]
            figures[path.stem] = dict(heights=[bar.get_height() for bar in axis.patches],
                hatches=[bar.get_hatch() for bar in axis.patches], text=[t.get_text() for t in axis.texts])
            return original(fig, path, plt, **kwargs)
        with tempfile.TemporaryDirectory() as temp, patch.object(reporting, "_save", side_effect=capture):
            result = screening_bar_outputs(rows, Path(temp), reference_time_s=143.)
            records = read_csv(result["time_to_internal_limit_comparison_csv"])
            self.assertEqual(records[0]["time_to_limit_s"], "130.0")
            self.assertEqual(records[1]["time_to_limit_s"], "")
            self.assertEqual(float(records[1]["time_to_limit_lower_bound_s"]), 200.)
            times = figures["time_to_internal_limit_comparison"]
            self.assertEqual(times["heights"], [130., 200.])
            self.assertEqual(times["hatches"], [None, "///"])
            self.assertIn(">200 s", times["text"])
            self.assertEqual(figures["peak_internal_temperature_comparison"]["heights"], [2000., 1700.])
            self.assertTrue(all(Path(path).is_file() for path in result.values()))
        self.assertIsNone(rows[1]["time_to_limit_s"])

    def test_bar_inputs_cannot_turn_an_unreached_limit_into_zero_time(self):
        row = dict(first_threshold_crossing([0, 200], [300, 1700]), design="nose", depth_m=.01)
        for changed in (dict(row, time_to_limit_s=0.), dict(row, peak_temperature_K=1900.),
                        dict(row, status="failure"), dict(row, duration_s=-1.)):
            with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
                screening_bar_outputs([changed], temp)
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(ValueError, "same depth"):
            screening_bar_outputs([row, dict(row, design="other", depth_m=.02)], temp)


if __name__ == "__main__":unittest.main()
