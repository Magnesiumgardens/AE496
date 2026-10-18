"""Recover verified native recordings without repeating CFD iterations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.aero import require_force_stability
from pipeline.force_history import (CSV_NAME, RAW_NAME, finish_history, force_history_diagnostics,
    assess_saved_force_history, recheck_saved_force_history)
from pipeline.utils import atomic_json, file_sha256
from test_force_history import NAMES, config, context, final_solver, native_file, rows


class ForceHistoryRecheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.cfg = config(self.work)
        self.raw = self.work / RAW_NAME
        native_file(self.raw, rows([100.] * 200, start=1201, step=1))
        self.solver = final_solver(0)
        metadata = finish_history(self.solver, self.cfg, context(self.cfg, self.raw))
        # Reproduce the former rejection of a generic report integer as the
        # final iteration counter, with raw proof retained by the old adapter.
        metadata.update(status="unavailable", final_iteration_verified=False,
                        error="Native force history extends past the measured final iteration",
                        force_stability={"status": "not_verified", "reason": "native adapter failed"})
        metadata.pop("source_sha256", None)
        metadata.pop("final_report_forces", None)
        metadata.pop("final_force_comparison", None)
        # The old counter check failed before writing any converted CSV.
        (self.work / CSV_NAME).unlink()
        for name in ("04_final.cas.h5", "04_final.dat.h5"):
            (self.work / name).write_bytes(("unchanged checkpoint " + name).encode())
        self.source = {"final_case": file_sha256(self.work / "04_final.cas.h5"),
                       "final_data": file_sha256(self.work / "04_final.dat.h5")}
        self.summary = dict(body_wall_zones=["body_wall"], force_history=metadata,
            numerical_quality={"status": "residual_screen_passed", "source_sha256": self.source})
        atomic_json(self.work / "summary.json", self.summary)

    def recheck(self):
        with patch("pipeline.aero.configure_force_reports", return_value=(NAMES, {"report_depth_m": 1.})):
            return recheck_saved_force_history(self.solver, self.cfg, ["body_wall"])

    def test_matching_checkpoint_and_native_final_forces_recover_without_iteration(self):
        result = self.recheck()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["final_iteration"], 1400)
        saved = json.loads((self.work / "summary.json").read_text())["force_history"]
        self.assertTrue(saved["rechecked_without_iterations"])
        self.assertEqual(saved["source_sha256"], self.source)
        self.assertEqual(saved["previous_verification"]["status"], "unavailable")
        self.assertEqual(saved["final_report_forces"]["drag_per_unit_span_N_m"], 100.)
        self.assertTrue(list((self.work / "solver_diagnostics").glob("force_history_before_recheck_*.json")))
        for key, name in (("final_case", "04_final.cas.h5"), ("final_data", "04_final.dat.h5")):
            self.assertEqual(file_sha256(self.work / name), self.source[key])
        self.assertEqual(assess_saved_force_history(self.cfg)["status"], "passed")

    def test_changed_pair_raw_or_conditions_cannot_be_relabelled_as_matching(self):
        original_raw = self.raw.read_bytes()
        original_pair = (self.work / "04_final.dat.h5").read_bytes()
        for changed in ("raw", "pair", "conditions"):
            with self.subTest(changed=changed):
                summary = copy.deepcopy(self.summary)
                self.raw.write_bytes(original_raw)
                (self.work / "04_final.dat.h5").write_bytes(original_pair)
                if changed == "raw":self.raw.write_text("unrelated native recording")
                if changed == "pair":(self.work / "04_final.dat.h5").write_bytes(b"different CFD data")
                if changed == "conditions":summary["force_history"]["conditions"]["flight"]["mach"] = 9.
                atomic_json(self.work / "summary.json", summary)
                with patch("pipeline.aero.configure_force_reports", side_effect=AssertionError("Do not query unrelated native forces")):
                    self.assertEqual(recheck_saved_force_history(self.solver, self.cfg, ["body_wall"])["status"], "not_verified")
                self.assertEqual(json.loads((self.work / "summary.json").read_text()), summary)

    def test_diagnostics_and_gate_preserve_the_underlying_recording_error(self):
        diagnosis = force_history_diagnostics(self.cfg)
        self.assertEqual(diagnosis["recording_status"], "unavailable")
        self.assertIn("extends past", diagnosis["force_stability"]["reason"])
        self.assertTrue(diagnosis["files"][RAW_NAME]["recorded_hash_matches"])
        self.assertIn('"Iteration"', diagnosis["files"][RAW_NAME]["first_lines"][1])
        self.cfg["aero"]["force_history"]["require_stable"] = True
        with self.assertRaisesRegex(ValueError, "extends past.*force-status"):
            require_force_stability(self.cfg, {"force_stability": diagnosis["force_stability"]})

    def test_mismatching_final_forces_remain_unverified_after_recheck(self):
        self.solver = final_solver(0, drag=130.)
        result = self.recheck()
        self.assertEqual(result["status"], "not_verified")
        self.assertIn("differs", result["reason"])
