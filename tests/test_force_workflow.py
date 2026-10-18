"""Force-screening failures retain measured data and never restart CFD implicitly."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from pipeline.aero import MODEL_VERSION
from pipeline.cfd import export_saved_case
from pipeline.config import load_config
from pipeline.study import (BatchError, output_hashes, run_case, stage_files,
                            stage_fingerprint)
from pipeline.temperature_export import temperature_paths
from pipeline.utils import atomic_json, file_sha256
from test_adiabatic_study import field_solver
from test_nose_trade import Reports


ROOT = Path(__file__).resolve().parents[1]
MONITOR = dict(enabled=True, sample_every_iterations=25, recent_samples=5,
               min_samples=5, relative_tolerance=.01,
               absolute_drag_tolerance_N_m=1e-6, require_stable=True)


class ForceWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work = self.root / "runs/blunt_1"
        self.work.mkdir(parents=True)
        self.cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        self.cfg["project"].update(name="blunt_1", work_dir=str(self.work))
        self.cfg["calibration"]["enabled"] = False
        geometry = self.root / "blunt_1.x_t"
        geometry.write_text("fixed shared CAD")
        self.cfg["geometry"]["input_file"] = str(geometry)
        self.cfg["aero"]["force_history"] = copy.deepcopy(MONITOR)
        self.config_path = self.work / "_batch/config.yaml"
        self.config_path.parent.mkdir()
        self.write_config()
        for name in ("02_mesh.msh.h5", "04_final.cas.h5", "04_final.dat.h5"):
            (self.work / name).write_bytes(("existing solved " + name).encode())
        self.pair = {name: file_sha256(self.work / name) for name in
                     ("04_final.cas.h5", "04_final.dat.h5")}

    def write_config(self):
        self.config_path.write_text(yaml.safe_dump(self.cfg))

    def solver(self):
        solver, _ = field_solver()
        solver.settings.file = SimpleNamespace(read_case_data=lambda **kwargs: None)
        solver.settings.setup.boundary_conditions.wall = Reports(body_wall=SimpleNamespace())
        self.exit_calls = []
        solver.exit = lambda: self.exit_calls.append(True)
        return solver

    @staticmethod
    def force_result(status):
        return dict(status="complete", model_version=MODEL_VERSION,
                    drag_per_unit_span_N_m=330., lift_per_unit_span_N_m=-327.,
                    pressure_drag_per_unit_span_N_m=320., viscous_drag_per_unit_span_N_m=10.,
                    component_status="complete", Cd=None, Cl=None,
                    force_stability=dict(status=status, reason="test measured drag screen"))

    def export(self, status):
        with patch("pipeline.cfd.launch_solver", return_value=self.solver()), \
                patch("pipeline.cfd.validate_saved_conditions"), \
                patch("pipeline.solution_quality.require_usable_solution",
                      return_value={"status": "residual_screen_passed"}), \
                patch("pipeline.cfd.aerodynamic_summary", return_value=self.force_result(status)):
            return export_saved_case(load_config(self.config_path))

    def assert_pair_retained(self):
        for name, digest in self.pair.items():
            self.assertEqual(file_sha256(self.work / name), digest)
        self.assertFalse((self.work / "_history").exists())

    def test_unstable_or_unverified_forces_retain_temperature_and_dimensional_exports(self):
        for status in ("failed", "not_verified", "insufficient_history"):
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValueError, "[Ff]orce|[Dd]rag"):
                    self.export(status)
                summary = json.loads((self.work / "aero_summary.json").read_text())
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["drag_per_unit_span_N_m"], 330.)
                self.assertEqual(summary["pressure_drag_per_unit_span_N_m"], 320.)
                self.assertEqual(summary["viscous_drag_per_unit_span_N_m"], 10.)
                self.assertEqual(summary["force_stability"]["status"], status)
                self.assertIsNone(summary["Cd"])
                self.assertEqual(summary["source_sha256"], {
                    "final_case": self.pair["04_final.cas.h5"],
                    "final_data": self.pair["04_final.dat.h5"]})
                csv_path, temperature_summary = temperature_paths(load_config(self.config_path))
                self.assertTrue(csv_path.is_file())
                self.assertEqual(json.loads(temperature_summary.read_text())["status"], "complete")
                self.assert_pair_retained()
                self.assertEqual(self.exit_calls, [True])

    def test_stable_force_screen_allows_export_without_creating_reference_width(self):
        result = self.export("passed")
        self.assertEqual(result["aerodynamics"]["force_stability"]["status"], "passed")
        self.assertIsNone(result["aerodynamics"]["Cd"])
        self.assertIsNone(self.cfg["aero"]["reference_area_per_unit_span_m"])
        self.assert_pair_retained()

    def test_explicit_diagnostic_opt_out_keeps_unverified_forces_available(self):
        self.cfg["aero"]["force_history"]["require_stable"] = False
        self.write_config()
        result = self.export("not_verified")
        self.assertEqual(result["aerodynamics"]["force_stability"]["status"], "not_verified")
        self.assert_pair_retained()

    def cached_outputs(self, force_status):
        cfg = load_config(self.config_path)
        files = stage_files(self.work, cfg)
        csv_path, summary_path = temperature_paths(cfg)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("wall_temperature_K,near_wall_fluid_temperature_K,segment_length_m\n2000,2001,1\n")
        atomic_json(summary_path, dict(status="complete", thermal_mode="radiative_equilibrium", face_count=1))
        atomic_json(self.work / "summary.json", dict(thermal_mode="radiative_equilibrium",
            source_geometry_sha256=file_sha256(cfg["geometry"]["input_file"]),
            source_mesh_sha256=file_sha256(self.work / "02_mesh.msh.h5")))
        aero = self.force_result(force_status)
        aero["source_sha256"] = {"final_case": self.pair["04_final.cas.h5"], "final_data": self.pair["04_final.dat.h5"]}
        atomic_json(self.work / "aero_summary.json", aero)
        records = {stage: dict(status="complete",
            fingerprint=stage_fingerprint(stage, cfg, {}, files),
            output_sha256=output_hashes(files[stage])) for stage in
            ("mesh", "solver", "temperature_export", "aero")}
        atomic_json(self.work / "_batch/stage_state.json", dict(version=4, strategy="single_solve", stages=records))

    def test_cached_unverified_aero_is_blocked_without_rerunning_or_archiving_cfd(self):
        self.cached_outputs("not_verified")
        calls = []
        with patch("pipeline.solution_quality.saved_quality", return_value={"status": "residual_screen_passed"}), \
                patch("pipeline.gas_reporting.temperature_complete", return_value=True), \
                patch("pipeline.gas_reporting.aero_complete_for_report", return_value=True):
            with self.assertRaisesRegex(BatchError, "[Ff]orce|[Dd]rag"):
                run_case(self.config_path, stop_after="aero", _runner=lambda *args, **kwargs: calls.append(args))
        self.assertEqual(calls, [])
        self.assert_pair_retained()
        self.assertEqual(json.loads((self.work / "aero_summary.json").read_text())[
            "drag_per_unit_span_N_m"], 330.)

    def test_monitor_settings_change_only_aero_fingerprint(self):
        old = copy.deepcopy(self.cfg)
        old["calibration"]["enabled"] = True
        old["aero"].pop("force_history")
        new = copy.deepcopy(old)
        new["aero"]["force_history"] = copy.deepcopy(MONITOR)
        files = stage_files(self.work, old)
        for paths in files.values():
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists(): path.write_text("unchanged upstream content")
        thermal = yaml.safe_load((ROOT / "thermal_analysis.yaml").read_text())
        for stage in ("mesh", "solver", "temperature_export", "calibration", "thermal"):
            with self.subTest(stage=stage):
                self.assertEqual(stage_fingerprint(stage, old, thermal, files),
                                 stage_fingerprint(stage, new, thermal, files))
        self.assertNotEqual(stage_fingerprint("aero", old, thermal, files),
                            stage_fingerprint("aero", new, thermal, files))

    def test_adopting_existing_aero_cannot_bypass_the_force_screen(self):
        self.cached_outputs("not_verified")
        path = self.work / "_batch/stage_state.json"
        state = json.loads(path.read_text())
        state["stages"].pop("aero")
        atomic_json(path, state)
        calls = []
        with patch("pipeline.solution_quality.saved_quality", return_value={"status": "residual_screen_passed"}), \
             patch("pipeline.gas_reporting.temperature_complete", return_value=True), \
             patch("pipeline.gas_reporting.aero_complete_for_report", return_value=True), \
             patch("pipeline.trade_study.aero_complete", return_value=True):
            with self.assertRaisesRegex(BatchError, "force-stability"):
                run_case(self.config_path, stop_after="aero", _runner=lambda *args, **kwargs: calls.append(args))
        self.assertFalse(calls)
        self.assert_pair_retained()


if __name__ == "__main__":
    unittest.main()
