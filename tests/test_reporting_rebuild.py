from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
from PIL import Image

from compare_nose_trade import compare
from pipeline.reporting import read_csv, summarize_wall_history, wall_coordinates

ROOT = Path(__file__).resolve().parents[1]


class ReportTests(unittest.TestCase):
    def test_normalization_preserves_samples_and_actual_distances(self):
        xy = np.array([[2, 0], [1, 1], [0, 1], [0, -1], [1, -1]])
        indices, distance, normalized = wall_coordinates(xy, [1, 0])
        self.assertEqual(indices.tolist(), [2, 1, 0, 4, 3])
        self.assertAlmostEqual(distance[-1], 2 + 2 * 2 ** 0.5)
        self.assertEqual(normalized[2], 0.5)
        self.assertEqual(normalized[[0, -1]].tolist(), [0, 1])
        records = [dict(time_s=t, surface_order_index=i, x_m=x, y_m=y,
                        wall_temperature_K=300 + 100 * i + t)
                   for t in (0, 10) for i, (x, y) in enumerate(xy)]
        history, profile, summary = summarize_wall_history(records, "nose")
        for record in profile:
            self.assertEqual(record["wall_temperature_K"], 310 + 100 * record["surface_order_index"])
        self.assertEqual(summary["final_nose_wall_temperature_K"], 310)
        self.assertEqual(history[0]["mean_wall_temperature_K"], 500)
        self.assertEqual(history[0]["mean_definition"], "simple")

    def test_ambiguous_nose_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rear closure"):
            wall_coordinates([[0, 0], [1, 1], [2, 0], [1, -1]])

    def test_real_reference_csvs_reproduce_comparisons_and_percase_reports(self):
        source = ROOT / "examples/reference_results"
        summary_before = (source / "temperature_comparison_summary.csv").read_bytes()
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            rows, created = compare(directory / "absent-runs", directory / "reports", source)
            self.assertEqual(rows, [])  # historical fixtures are not fresh simulations
            actual = read_csv(created["temperature_comparison_summary"])
            expected = read_csv(source / "temperature_comparison_summary.csv")
            for old, new in zip(expected, actual):
                for key in ("global_peak_wall_temperature_K", "final_max_wall_temperature_K", "final_mean_wall_temperature_K"):
                    self.assertEqual(float(old[key]), float(new[key]))
            original_profile = read_csv(source / "final_wall_temperature_normalized_position.csv")
            exported_profile = read_csv(created["final_wall_temperature_normalized_position"])
            for old, new in zip(original_profile, exported_profile):
                self.assertEqual(float(old["s_norm"]), float(new["s_norm"]))
                self.assertEqual(float(old["wall_temperature_K"]), float(new["wall_temperature_K"]))
            manifest = json.loads(Path(created["report_manifest"]).read_text())
            self.assertEqual(manifest["validated_unlabeled_wall_design"], "sharp")
            for name, color in (("base_geometry_negative_correct", (31, 119, 180)),
                                ("blunt_2", (255, 127, 14)), ("sharp", (44, 160, 44))):
                case = directory / "reports/cases" / name
                self.assertTrue((case / "final_wall_temperature.csv").is_file())
                with Image.open(case / "wall_temperature_history_comparison.png") as image:
                    pixels = np.array(image.convert("RGB"))
                self.assertTrue(np.any(np.all(pixels == color, axis=2)), name)
            self.assertTrue((directory / "reports/cases/sharp/wall_boundary.png").is_file())
            self.assertFalse((directory / "reports/cases/blunt_2/final_wall_heat_flux_map.png").exists())
            self.assertNotIn("Cd_by_geometry", created)
        self.assertEqual((source / "temperature_comparison_summary.csv").read_bytes(), summary_before)


if __name__ == "__main__":
    unittest.main()
