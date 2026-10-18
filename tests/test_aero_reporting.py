"""Force components and native-iteration diagnostics stay distinct from time."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from pipeline.aero import MODEL_VERSION, reference_state
from pipeline.gas_reporting import _read_force_history, create_gas_reports
from pipeline.reporting import build_result_row, read_csv
from pipeline.trade_study import aero_complete, aero_report_fields, create_trade_outputs, pareto_cases
from test_nose_trade import config


def metrics(status="passed"):
    return dict(status="complete", model_version=MODEL_VERSION, **reference_state(config()),
        body_wall_zones=["body_wall"], drag_per_unit_span_N_m=12., total_drag_per_unit_span_N_m=12.,
        pressure_drag_per_unit_span_N_m=10., viscous_drag_per_unit_span_N_m=2.,
        lift_per_unit_span_N_m=3., L_over_D=.25, Cd=None, Cd_pressure=None, Cd_viscous=None,
        component_status="complete", force_stability=dict(status=status))


class ForceReportingTests(unittest.TestCase):
    def test_flattened_components_keep_null_coefficients_and_stability(self):
        saved = metrics("failed")
        row = aero_report_fields(saved)
        self.assertEqual(row["pressure_drag_per_unit_span_N_m"], 10.)
        self.assertEqual(row["viscous_drag_per_unit_span_N_m"], 2.)
        self.assertEqual(row["drag_per_unit_span_N_m"], 12.)
        self.assertEqual(row["force_stability_status"], "failed")
        self.assertIsNone(row["Cd"])
        self.assertIsNone(row["Cd_pressure"])
        old = aero_report_fields(dict(drag_per_unit_span_N_m=5.))
        self.assertIsNone(old["pressure_drag_per_unit_span_N_m"])
        self.assertEqual(old["component_status"], "unavailable")
        self.assertEqual(old["force_stability_status"], "not_verified")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "aero_summary.json").write_text(json.dumps(saved))
            result = build_result_row("nose.x_t", "nose", root, "complete", 1.)
            self.assertEqual(result["force_stability_status"], "failed")
            self.assertEqual(result["pressure_drag_per_unit_span_N_m"], 10.)

    def test_pareto_retains_ties_but_excludes_unverified_and_unstable_forces(self):
        def row(name, drag, temperature, stability="passed"):
            return dict(case=name, status="complete", aero_status="complete",
                drag_per_unit_span_N_m=drag, peak_temperature_K=temperature, force_stability_status=stability,
                residual_status="residual_screen_passed")
        rows = [row("A", 10., 500.), row("B", 12., 400.), row("tie", 10., 500.),
                row("unverified", 1., 1., "not_verified"), row("failed", 1., 1., "failed"),
                row("short", 1., 1., "insufficient_history")]
        self.assertEqual(pareto_cases(rows), ["A", "B", "tie"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row_ in rows:
                case = root / row_["case"]; case.mkdir()
                (case / "aero_summary.json").write_text(json.dumps(metrics(row_["force_stability_status"])))
            created = create_trade_outputs(root, rows, root / "comparison")
            report = json.loads(Path(created["nose_trade_screening"]).read_text())
            self.assertEqual(report["pareto_cases"], ["A", "B", "tie"])
            self.assertEqual({r["case"] for r in report["cases_excluded_from_pareto_for_force_checks"]},
                             {"unverified", "failed", "short"})
            self.assertNotIn("Cd_by_geometry", created)

    def test_stationary_drag_does_not_accept_an_unscreened_residual_solution(self):
        candidate = dict(case="apparently_stable", status="complete", aero_status="complete",
                         drag_per_unit_span_N_m=10., peak_temperature_K=500.,
                         force_stability_status="passed", residual_status="not_verified")
        self.assertEqual(pareto_cases([candidate]), [])
        candidate["residual_status"] = "residual_screen_passed"
        self.assertEqual(pareto_cases([candidate]), ["apparently_stable"])

    def test_current_history_checks_invalidate_cached_status_without_hiding_unverified_force_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = config(); cfg["project"] = {"work_dir": str(root)}
            for name in ("04_final.cas.h5", "04_final.dat.h5"):
                (root / name).write_bytes(name.encode())
            summary = metrics("not_verified")
            (root / "aero_summary.json").write_text(json.dumps(summary))
            with patch("pipeline.force_history.assess_saved_force_history", return_value=dict(status="not_verified")):
                self.assertTrue(aero_complete(root, cfg))
            with patch("pipeline.force_history.assess_saved_force_history", return_value=dict(status="passed")):
                self.assertFalse(aero_complete(root, cfg))

    def test_force_reports_are_native_iteration_diagnostics_without_temperature_export(self):
        history = [dict(iteration=25, drag_per_unit_span_N_m=12.2, lift_per_unit_span_N_m=3.1),
                   dict(iteration=100, drag_per_unit_span_N_m=12., lift_per_unit_span_N_m=3.)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); work = root / "nose"; (work / "_batch").mkdir(parents=True)
            cfg = config(); cfg.update(project=dict(name="nose", work_dir=str(work)),
                                       thermal=dict(mode="adiabatic"), mesh={})
            (work / "_batch/config.yaml").write_text(yaml.safe_dump(cfg))
            (work / "aero_summary.json").write_text(json.dumps(metrics("failed")))
            with patch("pipeline.gas_reporting.temperature_complete", return_value=False), \
                 patch("pipeline.gas_reporting.aero_complete_for_report", return_value=True), \
                 patch("pipeline.force_history.read_saved_force_history", return_value=(history, {})):
                rows, created = create_gas_reports(root, root / "comparison")
            self.assertEqual(rows[0]["force_stability_status"], "failed")
            self.assertEqual(rows[0]["pressure_drag_per_unit_span_N_m"], 10.)
            self.assertIsNone(rows[0]["Cd"])
            values = read_csv(created["force_iteration_history_comparison_csv"])
            self.assertEqual([int(v["iteration"]) for v in values], [25, 100])
            self.assertNotIn("time_s", values[0])
            self.assertTrue((work / "reports/force_iteration_history.png").is_file())
            self.assertTrue(Path(created["drag_components_comparison"]).is_file())
            self.assertTrue(Path(created["force_iteration_history_comparison"]).is_file())
            with patch("pipeline.gas_reporting.temperature_complete", return_value=False), \
                 patch("pipeline.gas_reporting.aero_complete_for_report", return_value=True), \
                 patch("pipeline.force_history.read_saved_force_history", side_effect=ValueError("Changed source hash")):
                _, fresh = create_gas_reports(root, root / "comparison")
            self.assertFalse((work / "reports/force_iteration_history.png").exists())
            self.assertNotIn("force_iteration_history_comparison", fresh)
            self.assertEqual(read_csv(fresh["force_iteration_history_comparison_csv"]), [])

    def test_stale_native_history_is_not_plotted(self):
        cfg = config(); cfg.update(project=dict(work_dir="unused"), thermal=dict(mode="adiabatic"))
        warnings = []
        with patch("pipeline.force_history.read_saved_force_history", side_effect=ValueError("Changed source hash")):
            self.assertEqual(_read_force_history(cfg, warnings, "nose"), [])
        self.assertIn("Changed source hash", warnings[0])


if __name__ == "__main__":
    unittest.main()
