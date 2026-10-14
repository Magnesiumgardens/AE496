"""Continue one saved CFD endpoint without losing the shared mesh or other state."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from pipeline.config import load_config
from pipeline.continuation import continue_case, source_hashes
from pipeline.endpoint_workflow import effective_cfg, fingerprint, stage_files
from pipeline.study import BatchError, reusable
from pipeline.temperature_export import temperature_conditions
from pipeline.utils import atomic_json, file_sha256

import test_endpoint_workflow as workflow_fixtures


class EndpointContinuationTests(unittest.TestCase):
    def setUp(self):
        # Reuse the real endpoint field/export adapters and virtual subprocess
        # fixture without inheriting and rediscovering its independent tests.
        self.fixture = workflow_fixtures.EndpointWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.study(stop_after="aero")
        self.fixture.calls.clear()
        self.commands = []

    def runner(self, command, log_path, **kwargs):
        f = self.fixture
        self.commands.append(command)
        cfg = load_config(command[command.index("--config") + 1])
        kind = cfg["_endpoint_kind"]
        active = f.work / "cfd" / kind
        candidate = Path(command[command.index("--output-dir") + 1])
        self.assertEqual(Path(cfg["project"]["work_dir"]), active)
        self.assertEqual(Path(cfg["_study_work_dir"]), f.work)
        self.assertTrue((f.work / "02_mesh.msh.h5").is_file())
        self.assertFalse((active / "02_mesh.msh.h5").exists())
        self.assertIn(active / "_continuation", candidate.parents)
        self.assertIn("--continue-saved", command)
        self.assertNotIn("--stage", command)
        for name in ("04_final.cas.h5", "04_final.dat.h5"):
            (candidate / name).write_bytes(("continued " + kind + name).encode())
        summary = copy.deepcopy(json.loads((active / "summary.json").read_text()))
        summary.pop("numerical_quality", None)
        summary.update(source_conditions=temperature_conditions(cfg),
                       source_mesh_sha256=file_sha256(f.work / "02_mesh.msh.h5"),
                       source_geometry_sha256=file_sha256(cfg["geometry"]["input_file"]),
                       temperature_history={"status":"disabled"},
                       continuation={"source_sha256":source_hashes(active),
                                     "initialization_performed":False,
                                     "additional_iteration_budget":int(command[command.index("--iterations") + 1])})
        atomic_json(candidate / "summary.json", summary)
        log_path.write_text(workflow_fixtures.GOOD_LOG)

    def state(self):
        return json.loads((self.fixture.work / "_batch/stage_state.json").read_text())

    def test_cold_continuation_preserves_common_mesh_and_resumes_after_archiving_reference_and_loads(self):
        f = self.fixture
        cold = f.work / "cfd/cold_wall"
        old_cold = source_hashes(cold)
        old_reference = (f.work / "cfd/adiabatic/04_final.dat.h5").read_bytes()
        old_loads = (f.work / "thermal_bc/convection_nodes.csv").read_bytes()
        mesh_bytes = (f.work / "02_mesh.msh.h5").read_bytes()
        mesh_record = copy.deepcopy(self.state()["stages"]["mesh"])
        result = continue_case(f.path, 50, courant=.35, endpoint="cold_wall", _runner=self.runner)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["work_dir"], str(f.work))
        self.assertEqual((f.work / "02_mesh.msh.h5").read_bytes(), mesh_bytes)
        self.assertEqual(self.state()["stages"]["mesh"], mesh_record)
        summary = json.loads((cold / "summary.json").read_text())
        self.assertEqual(summary["continuation"]["source_sha256"], old_cold)
        self.assertEqual(summary["numerical_quality"]["source_sha256"], source_hashes(cold))
        self.assertEqual(self.state()["stages"]["cold_wall"]["status"], "complete")
        self.assertFalse((f.work / "cfd/adiabatic/04_final.dat.h5").exists())
        self.assertFalse((f.work / "thermal_bc/convection_nodes.csv").exists())
        self.assertFalse((f.work / "aero_summary.json").exists())
        reference_archives = list((f.work / "_history").glob("*/cfd/adiabatic/04_final.dat.h5"))
        loads_archives = list((f.work / "_history").glob("*/thermal_bc/convection_nodes.csv"))
        self.assertEqual([p.read_bytes() for p in reference_archives], [old_reference])
        self.assertEqual([p.read_bytes() for p in loads_archives], [old_loads])
        self.assertEqual(source_hashes(Path(result["archive"])), old_cold)
        # Continue produces a valid cold cache record; a normal run must not
        # reset that newly developed field or recreate the common mesh.
        f.study(stop_after="aero")
        self.assertEqual(f.calls, ["cold_export", "adiabatic", "adiabatic_export", "thermal_bc", "aero"])
        new_reference = json.loads((f.work / "cfd/adiabatic/summary.json").read_text())
        self.assertEqual(new_reference["reference_source_sha256"], source_hashes(cold))

    def test_adiabatic_courant_override_preserves_cold_numerics_and_rebuilds_only_derived_products(self):
        f = self.fixture
        cfg = load_config(f.path)
        cold_cfg = effective_cfg(cfg, "cold_wall")
        cold = f.work / "cfd/cold_wall"
        old_cold = source_hashes(cold)
        old_adiabatic = source_hashes(f.work / "cfd/adiabatic")
        mesh_record = copy.deepcopy(self.state()["stages"]["mesh"])
        cold_record = copy.deepcopy(self.state()["stages"]["cold_wall"])
        result = continue_case(f.path, 75, courant=.45, endpoint="adiabatic", _runner=self.runner)
        self.assertEqual(result["status"], "complete")
        updated = load_config(f.path)
        self.assertEqual(updated["workflow"]["adiabatic_solver"]["courant_number"], .45)
        self.assertEqual(updated["solver"], cfg["solver"])
        self.assertEqual(effective_cfg(updated, "cold_wall")["solver"], cold_cfg["solver"])
        self.assertEqual(source_hashes(cold), old_cold)
        self.assertEqual(self.state()["stages"]["mesh"], mesh_record)
        self.assertEqual(self.state()["stages"]["cold_wall"], cold_record)
        reference = json.loads((f.work / "cfd/adiabatic/summary.json").read_text())
        self.assertEqual(reference["reference_source_sha256"], old_cold)
        self.assertEqual(reference["continuation"]["source_sha256"], old_adiabatic)
        self.assertEqual(source_hashes(Path(result["archive"])), old_adiabatic)
        self.assertFalse((f.work / "temperature/wall_temperatures.csv").exists())
        self.assertFalse((f.work / "thermal_bc/convection_nodes.csv").exists())
        self.assertFalse((f.work / "aero_summary.json").exists())
        f.study(stop_after="aero")
        self.assertEqual(f.calls, ["adiabatic_export", "thermal_bc", "aero"])

    def test_locked_downstream_export_preserves_promoted_checkpoint_and_reusable_parent_record(self):
        f = self.fixture
        cold = f.work / "cfd/cold_wall"
        original = source_hashes(cold)
        with patch("pipeline.endpoint_workflow.archive", side_effect=OSError("locked export")):
            with self.assertRaisesRegex(OSError, "locked export"):
                continue_case(f.path, 50, endpoint="cold_wall", _runner=self.runner)
        current = source_hashes(cold)
        self.assertNotEqual(current, original)
        summary = json.loads((cold / "summary.json").read_text())
        self.assertEqual(summary["numerical_quality"]["status"], "residual_screen_passed")
        self.assertEqual(summary["numerical_quality"]["source_sha256"], current)
        cfg = load_config(f.path)
        files = stage_files(cfg)
        record = self.state()["stages"]["cold_wall"]
        self.assertTrue(reusable(record, fingerprint("cold_wall", cfg, {}, files), files["cold_wall"]))
        self.assertEqual(self.state()["stages"]["adiabatic"]["status"], "stale")
        backups = list((cold / "_history").glob("*_continue"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(source_hashes(backups[0]), original)
        # Once the lock clears, recovery can reuse the promoted solve; a user
        # may inspect its checkpoint without triggering a fresh cold startup.
        result = f.study(stop_after="cold_wall")
        self.assertEqual(result["stages"]["cold_wall"], "reused")
        self.assertFalse(f.calls)

    def test_adiabatic_continuation_rejects_changed_or_unrecorded_cold_reference_before_launch(self):
        f = self.fixture
        cold_data = f.work / "cfd/cold_wall/04_final.dat.h5"
        previous = cold_data.read_bytes()
        cold_data.write_bytes(b"different cold-wall checkpoint")
        with self.assertRaisesRegex(BatchError, "current cold-wall reference checkpoint"):
            continue_case(f.path, 50, endpoint="adiabatic", _runner=self.runner)
        self.assertFalse(self.commands)
        cold_data.write_bytes(previous)
        summary_path = f.work / "cfd/adiabatic/summary.json"
        summary = json.loads(summary_path.read_text())
        summary.pop("reference_source_sha256")
        atomic_json(summary_path, summary)
        with self.assertRaisesRegex(BatchError, "current cold-wall reference checkpoint"):
            continue_case(f.path, 50, endpoint="adiabatic", _runner=self.runner)
        self.assertFalse(self.commands)


if __name__ == "__main__":
    unittest.main()
