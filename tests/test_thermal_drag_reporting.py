"""Saved-source matching and distinct signed heat/drag comparison metrics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import thermal_drag_reporting as reports
from pipeline.reporting import read_csv, write_csv


class ThermalDragTests(unittest.TestCase):
    def fixture(self, root):
        output = root / "comparison"
        output.mkdir()
        sources = {"final_case": "case-hash", "final_data": "data-hash"}
        for name, drag in (("blunt", 123.), ("sharp", 456.)):
            work = root / name
            (work / "_batch").mkdir(parents=True)
            (work / "_batch/config.yaml").write_text(f"project: {{name: {name}}}\n")
            (work / "thermal_analysis").mkdir()
            (work / "thermal_analysis/thermal_summary.json").write_text(json.dumps(dict(source_sha256=sources)))
            (work / "aero_summary.json").write_text(json.dumps(dict(drag_per_unit_span_N_m=drag,
                source_sha256=sources, force_stability=dict(status="passed" if name == "blunt" else "not_verified"),
                residual_status="residual_screen_passed")))
        write_csv(output / "total_heat_load_comparison.csv", [
            dict(design="sharp", final_time_s=200, final_net_heat_load_per_unit_span_J_m=800),
            dict(design="blunt", final_time_s=200, final_net_heat_load_per_unit_span_J_m=-200)])
        write_csv(output / "peak_heat_flux_comparison.csv", [
            dict(design="blunt", final_time_s=200, peak_net_heat_flux_W_m2=1000,
                 time_of_peak_net_flux_s=0, peak_net_surface_order_index=5),
            dict(design="sharp", final_time_s=200, peak_net_heat_flux_W_m2=2000,
                 time_of_peak_net_flux_s=20, peak_net_surface_order_index=7)])
        return output

    def test_pairs_by_case_retains_signed_energy_units_and_force_check_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = self.fixture(root)
            saved = {}
            original = reports._save
            def capture(fig, path, plt):
                saved[path.stem] = [collection.get_offsets().tolist() for collection in fig.axes[0].collections]
                return original(fig, path, plt)
            with patch("pipeline.mechanical_reporting.mechanical_complete", return_value=True), \
                 patch("pipeline.gas_reporting.aero_complete_for_report", return_value=True), \
                 patch.object(reports, "_save", side_effect=capture):
                result = reports.thermal_drag_outputs(root, ["blunt", "sharp"], output,
                                                       {"blunt": "blue", "sharp": "orange"})
            self.assertEqual(saved["final_heat_load_vs_drag"], [[[123., -200.]], [[456., 800.]]])
            self.assertEqual(saved["peak_heat_flux_vs_drag"], [[[123., 1000.]], [[456., 2000.]]])
            rows = read_csv(result["peak_heat_flux_vs_drag_csv"])
            self.assertEqual(float(rows[0]["final_time_s"]), 200.)
            self.assertEqual(float(rows[0]["time_of_peak_net_flux_s"]), 0.)
            self.assertEqual(rows[0]["force_checks_passed"], "True")
            self.assertEqual(rows[1]["force_checks_passed"], "False")
            self.assertTrue(all(Path(path).is_file() for path in result.values()))

    def test_missing_stale_or_mismatched_aero_does_not_enter_the_trade(self):
        for fault in ("thermal", "aero", "source", "drag"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                output = self.fixture(root)
                path = root / "blunt/aero_summary.json"
                aero = json.loads(path.read_text())
                if fault == "source":aero["source_sha256"]["final_data"] = "other-case"
                if fault == "drag":aero["drag_per_unit_span_N_m"] = None
                path.write_text(json.dumps(aero))
                for stem in ("final_heat_load_vs_drag", "peak_heat_flux_vs_drag"):
                    (output / (stem + ".png")).write_bytes(b"stale plot")
                with patch("pipeline.mechanical_reporting.mechanical_complete", return_value=fault != "thermal"), \
                     patch("pipeline.gas_reporting.aero_complete_for_report", return_value=fault != "aero"):
                    result = reports.thermal_drag_outputs(root, ["blunt"], output, {"blunt": "blue"})
                pairing = read_csv(result["thermal_drag_pairing_csv"])
                self.assertEqual(pairing[0]["included"], "False")
                self.assertTrue(pairing[0]["reason"])
                self.assertFalse((output / "final_heat_load_vs_drag.png").exists())
                self.assertFalse((output / "peak_heat_flux_vs_drag.png").exists())


if __name__ == "__main__":unittest.main()
