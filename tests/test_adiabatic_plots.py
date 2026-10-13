"""Figure freshness and scientific interpretation of adiabatic CFD reports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from pipeline.gas_reporting import _read_history, build_gas_row, create_gas_reports, temperature_complete
from pipeline.reporting import write_csv
from pipeline.utils import file_sha256
from pipeline.temperature_export import temperature_conditions
import test_adiabatic_study as fixtures


class AdiabaticPlotTests(unittest.TestCase):
    def save_case(self, work):
        return fixtures.GasReportTests().save_case(work)

    def test_individual_and_comparison_figures_use_exported_gas_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_case(root / "blunt")
            self.save_case(root / "sharp")
            with patch("pipeline.cfd.launch_solver", side_effect=AssertionError("Offline plots must not launch Fluent")):
                rows, created = create_gas_reports(root, root / "comparison")
            self.assertEqual([row["case"] for row in rows], ["blunt", "sharp"])
            for key in ("temperature_statistics_comparison", "wall_temperature_K_maps_comparison",
                        "near_wall_fluid_temperature_K_maps_comparison", "drag_vs_temperature"):
                self.assertTrue(Path(created[key]).is_file())
            report = json.loads((root / "comparison/adiabatic_report.json").read_text())
            self.assertEqual(report["temperature_color_ranges_K"]["wall_temperature_K"], [2000, 2300])
            self.assertIn("No physical-time history", report["history"])
            self.assertIn("no invented", report["spatial_method"])
            for name in ("blunt", "sharp"):
                outputs = report["case_files"][name]
                for key in ("fluid_temperature_near_wall",
                            "adiabatic_wall_temperature_map", "near_wall_fluid_temperature_map",
                            "wall_temperature_vs_position", "wall_boundary"):
                    self.assertTrue(Path(outputs[key]).is_file())
                case_manifest = json.loads((root / name / "reports/adiabatic_case_report.json").read_text())
                self.assertEqual(case_manifest["temperature_color_ranges_K"], report["temperature_color_ranges_K"])
                self.assertEqual(case_manifest["wall_boundary_condition"], "q_wall = 0; no radiation or imposed wall temperature")
                self.assertFalse((root / name / "reports/temperature_history.png").exists())

    def test_changed_conditions_remove_stale_comparisons_but_keep_case_maps_and_historical_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_case(root / "blunt")
            self.save_case(root / "sharp")
            create_gas_reports(root, root / "comparison")
            historic = root / "comparison/wall_temperature_history_comparison.png"
            historic.write_bytes(b"historical reference")
            cfg_path = root / "sharp/_batch/config.yaml"
            cfg = yaml.safe_load(cfg_path.read_text())
            cfg["mesh"]["global_max_size"] *= 2
            cfg_path.write_text(yaml.safe_dump(cfg))
            # Represent a new verified export made with the changed mesh settings.
            summary_path = root / "sharp/temperature/temperature_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["conditions"] = temperature_conditions(cfg)
            summary_path.write_text(json.dumps(summary))
            rows, created = create_gas_reports(root, root / "comparison")
            manifest = json.loads((root / "comparison/adiabatic_report.json").read_text())
            self.assertFalse(manifest["comparable_conditions"])
            self.assertNotIn("wall_temperature_K_comparison", created)
            self.assertFalse((root / "comparison/wall_temperature_K_comparison.png").exists())
            self.assertFalse((root / "comparison/drag_vs_near_wall_temperature.png").exists())
            self.assertTrue((root / "sharp/reports/adiabatic_wall_temperature_map.png").exists())
            self.assertEqual(historic.read_bytes(), b"historical reference")

    def test_unavailable_optional_nodes_remove_old_maps_without_losing_wall_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_case(root / "nose")
            create_gas_reports(root, root / "comparison")
            summary_path = root / "nose/temperature/temperature_summary.json"
            summary = json.loads(summary_path.read_text())
            summary["fluid_nodes"] = {"status": "unavailable", "error": "Fluent omitted interior data"}
            summary_path.write_text(json.dumps(summary))
            create_gas_reports(root, root / "comparison")
            self.assertFalse((root / "nose/reports/fluid_temperature_nodes.png").exists())
            self.assertFalse((root / "nose/reports/fluid_temperature_near_wall.png").exists())
            self.assertTrue((root / "nose/reports/temperature_profile.png").exists())
            manifest = json.loads((root / "comparison/adiabatic_report.json").read_text())
            self.assertTrue(any("Fluent omitted interior data" in value for value in manifest["warnings"]))

    def test_stale_solution_removes_current_gas_figures_but_retains_historical_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_case(root / "nose")
            create_gas_reports(root, root / "comparison")
            historical = root / "nose/reports/final_2d_temperature_field.png"
            historical.write_bytes(b"saved solid result")
            (root / "nose/04_final.dat.h5").write_bytes(b"new unsampled solution")
            rows, created = create_gas_reports(root, root / "comparison")
            self.assertEqual(rows[0]["temperature_status"], "missing_or_stale")
            self.assertFalse((root / "nose/reports/temperature_profile.png").exists())
            self.assertFalse((root / "comparison/wall_temperature_K_comparison.png").exists())
            self.assertEqual(historical.read_bytes(), b"saved solid result")

    def test_invalid_reference_settings_produce_failed_row_instead_of_crashing_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.save_case(root / "nose")
            cfg["flight"]["mach"] = "not a number"
            row = build_gas_row(cfg, status="failed", error="solver validation failed")
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["aero_status"], "missing_or_stale")
            self.assertIn("solver validation failed", row["error"])
            self.assertIn("flight.mach", row["error"])
            (root / "nose/_batch/config.yaml").write_text(yaml.safe_dump(cfg))
            rows, _ = create_gas_reports(root, root / "comparison")
            self.assertEqual(rows[0]["status"], "failed")

    def test_changed_export_sampling_settings_mark_package_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.save_case(Path(directory) / "nose")
            self.assertTrue(temperature_complete(cfg))
            cfg["temperature_export"]["topology_tolerance_m"] = 5e-10
            self.assertFalse(temperature_complete(cfg))

    def test_changed_flow_or_physics_or_mesh_marks_temperature_package_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.save_case(Path(directory) / "nose")
            self.assertTrue(temperature_complete(cfg))
            cfg["flight"]["mach"] += 1
            self.assertFalse(temperature_complete(cfg))

    def test_iteration_convergence_uses_provenance_and_removes_stale_history_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_case(root / "nose")
            work = root / "nose"
            history = work / "solver_temperature_history.csv"
            write_csv(history, [dict(iteration=100, wall_temperature_min_k=500,
                                    wall_temperature_avg_k=1400, wall_temperature_max_k=2100),
                                dict(iteration=200, wall_temperature_min_k=520,
                                    wall_temperature_avg_k=1450, wall_temperature_max_k=2150)])
            source = json.loads((work / "temperature/temperature_summary.json").read_text())["source_sha256"]
            summary = {"thermal_mode": "adiabatic", "temperature_history": {
                "status": "complete", "csv_sha256": file_sha256(history), "source_sha256": source}}
            (work / "summary.json").write_text(json.dumps(summary))
            _, created = create_gas_reports(root, root / "comparison")
            self.assertTrue(Path(created["temperature_convergence_comparison"]).is_file())
            manifest = json.loads((work / "reports/adiabatic_case_report.json").read_text())
            self.assertEqual(manifest["convergence_history_samples"], 2)
            self.assertTrue(Path(manifest["files"]["temperature_convergence"]).is_file())
            summary["temperature_history"]["source_sha256"] = {"final_case": "old", "final_data": "old"}
            (work / "summary.json").write_text(json.dumps(summary))
            _, created = create_gas_reports(root, root / "comparison")
            self.assertNotIn("temperature_convergence_comparison", created)
            self.assertFalse((work / "reports/temperature_convergence.png").exists())
            self.assertFalse((root / "comparison/temperature_convergence_comparison.png").exists())
            manifest = json.loads((root / "comparison/adiabatic_report.json").read_text())
            self.assertTrue(any("does not belong" in warning for warning in manifest["warnings"]))

    def test_unusable_optional_history_warns_without_invalidating_temperature_package(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "nose"
            cfg = self.save_case(work)
            temperature_summary = json.loads((work / "temperature/temperature_summary.json").read_text())
            history = work / "solver_temperature_history.csv"
            for samples in ([dict(iteration=200, wall_temperature_min_k=500,
                                  wall_temperature_avg_k=1400, wall_temperature_max_k=2100)],
                            [dict(iteration=100, wall_temperature_min_k=500,
                                  wall_temperature_avg_k=1400, wall_temperature_max_k=2100),
                             dict(iteration=200, wall_temperature_min_k=500,
                                  wall_temperature_avg_k=2200, wall_temperature_max_k=2100)]):
                write_csv(history, samples)
                (work / "summary.json").write_text(json.dumps({"thermal_mode": "adiabatic", "temperature_history": {
                    "status": "partial", "csv_sha256": file_sha256(history),
                    "source_sha256": temperature_summary["source_sha256"]}}))
                warnings = []
                self.assertEqual(_read_history(work, temperature_summary, warnings, "nose"), [])
                self.assertTrue(warnings)
                self.assertTrue(temperature_complete(cfg))


if __name__ == "__main__":
    unittest.main()
