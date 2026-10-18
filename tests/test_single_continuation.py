"""Radiating saved-flow continuation and Windows archive recovery."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline.config import load_config
from pipeline.continuation import continue_case, continue_solver, source_hashes
from pipeline.study import BatchError, reusable, stage_files, stage_fingerprint
from pipeline.utils import atomic_json
import test_single_workflow as workflow_fixtures


class SingleContinuationTests(unittest.TestCase):
    def setUp(self):
        # Compose the fixture without inheriting and duplicating its tests.
        self.fixture = workflow_fixtures.SingleWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.study()
        self.fixture.calls.clear()
        self.commands = []

    def state(self):
        return json.loads((self.fixture.work / "_batch/stage_state.json").read_text())

    def runner(self, command, log_path, **kwargs):
        self.commands.append(command)
        candidate = Path(command[command.index("--output-dir") + 1])
        work = self.fixture.work
        for name in ("04_final.cas.h5", "04_final.dat.h5"):
            (candidate / name).write_bytes(("continued " + name).encode())
        summary = copy.deepcopy(json.loads((work / "summary.json").read_text()))
        summary.pop("numerical_quality", None)
        summary.update(temperature_history={"status": "disabled"},
                       continuation={"source_sha256": source_hashes(work),
                                     "initialization_performed": False})
        atomic_json(candidate / "summary.json", summary)
        log_path.write_text(workflow_fixtures.GOOD_LOG)

    def test_locked_export_does_not_reset_promoted_radiating_solution(self):
        f = self.fixture
        original = source_hashes(f.work)
        old_export = (f.work / "temperature/wall_temperatures.csv").read_bytes()
        with patch("pipeline.study._archive_stage", side_effect=OSError("locked export")):
            with self.assertRaisesRegex(OSError, "locked export"):
                continue_case(f.path, 50, _runner=self.runner)
        current = source_hashes(f.work)
        self.assertNotEqual(current, original)
        summary = json.loads((f.work / "summary.json").read_text())
        self.assertEqual(summary["numerical_quality"]["status"], "residual_screen_passed")
        self.assertEqual(summary["numerical_quality"]["source_sha256"], current)
        cfg = load_config(f.path)
        files = stage_files(f.work, cfg)
        state = self.state()
        self.assertTrue(reusable(state["stages"]["solver"],
                                 stage_fingerprint("solver", cfg, {}, files), files["solver"]))
        for stage in ("temperature_export", "aero", "thermal"):
            self.assertEqual(state["stages"][stage]["status"], "stale")
        self.assertEqual((f.work / "temperature/wall_temperatures.csv").read_bytes(), old_export)
        backups = list((f.work / "_history").glob("*_continue"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(source_hashes(backups[0]), original)
        # The lock clearing must resume only downstream work, never initialize
        # a fresh flow in place of the promoted, screened checkpoint.
        result = f.study()
        self.assertEqual(result["stages"]["solver"], "reused")
        self.assertEqual(f.calls, ["export_temperature_only.py", "export_aero_only.py",
                                   "run_thermal_analysis.py"])

    def test_native_continuation_preserves_flow_and_invalidates_old_wall_statistics(self):
        f = self.fixture
        cfg = load_config(f.path)
        original = source_hashes(f.work)
        metadata = json.loads((f.work / "summary.json").read_text())
        for key in ("wall_temperature_min_k", "wall_temperature_avg_k", "wall_temperature_max_k"):
            metadata[key] = 300.0
        metadata["wall_temperature_report"] = {"status": "complete"}
        atomic_json(f.work / "summary.json", metadata)
        calls = []

        def forbidden(*args, **kwargs):
            self.fail("Saved-flow continuation must not read_mesh or initialize")

        def write_case(file_name):
            calls.append("write_case_data")
            Path(file_name).write_bytes(b"native continued case")
            Path(file_name.replace(".cas.h5", ".dat.h5")).write_bytes(b"native continued data")

        solver = SimpleNamespace(
            exit=lambda: calls.append("exit"),
            settings=SimpleNamespace(
                file=SimpleNamespace(read_case_data=lambda **kwargs: calls.append("read_case_data"),
                                     read_mesh=forbidden, write_case_data=write_case),
                setup=SimpleNamespace(general=SimpleNamespace(solver=SimpleNamespace(
                    time=SimpleNamespace(get_state=lambda: "steady")))),
                mesh=SimpleNamespace(quality=lambda: calls.append("mesh_quality")),
                solution=SimpleNamespace(initialization=SimpleNamespace(
                    hybrid_initialize=forbidden, standard_initialize=forbidden)),
            ),
        )

        def iterate(engine, segment, zones):
            self.assertIs(engine, solver)
            self.assertEqual(segment["solver"]["max_iterations"], 50)
            self.assertEqual(zones, ["body_wall"])
            calls.append("iterate")
            return {"temperature_history": {"status": "disabled"}, "force_history": {"status": "disabled"}}

        staged = f.work / "_continuation/native-test"
        with patch("pipeline.continuation.launch_solver", return_value=solver), \
             patch("pipeline.continuation.validate_saved_conditions") as validate, \
             patch("pipeline.continuation.configure_numerics", return_value={}), \
             patch("pipeline.continuation.configure_residual_criteria", return_value={}), \
             patch("pipeline.continuation.iterate_with_histories", side_effect=iterate):
            summary = continue_solver(cfg, 50, staged)
        validate.assert_called_once_with(solver, cfg)
        self.assertEqual(calls, ["read_case_data", "mesh_quality", "iterate", "write_case_data", "exit"])
        self.assertFalse(summary["continuation"]["initialization_performed"])
        self.assertEqual(summary["continuation"]["source_sha256"], original)
        self.assertEqual(summary["thermal_mode"], "radiative_equilibrium")
        self.assertEqual(summary["wall_temperature_report"]["status"], "pending_export")
        self.assertNotIn("numerical_quality", summary)
        for key in ("wall_temperature_min_k", "wall_temperature_avg_k", "wall_temperature_max_k"):
            self.assertNotIn(key, summary)
        self.assertEqual(source_hashes(f.work), original)

    def test_direct_native_continuation_checks_cad_mesh_and_pair_before_launch(self):
        f = self.fixture
        cfg = load_config(f.path)
        paths = [Path(cfg["geometry"]["input_file"]), f.work / "02_mesh.msh.h5",
                 f.work / "04_final.cas.h5", f.work / "04_final.dat.h5"]
        for index, path in enumerate(paths):
            with self.subTest(path=path.name):
                previous = path.read_bytes()
                try:
                    path.write_bytes(b"changed checkpoint input")
                    with patch("pipeline.continuation.launch_solver") as launch:
                        with self.assertRaises(BatchError):
                            continue_solver(cfg, 50, f.work / "_continuation" / str(index))
                        launch.assert_not_called()
                finally:
                    path.write_bytes(previous)


if __name__ == "__main__":
    unittest.main()
