"""Geometric selection is independent of failed heating coefficients."""
from pathlib import Path
import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

import run_calibration
from pipeline.aerothermal_calibration import calibration_cfg, calibration_paths
from pipeline.migration import single_solve_config, single_solve_state
from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.thermal_scope import active_nodes, active_segments, scoped_face_values_to_nodes
from pipeline.utils import file_sha256
from test_calibration_runner import runner_case


class ThermalScopeTests(unittest.TestCase):
    # The split rear chain 2->3->4->0 crosses the stored loop seam.
    xy = np.array([[0., .1], [1., 0.], [0., -.1], [0., -.05], [0., 0.]])

    def test_split_rear_wraps_closed_loop_and_keeps_corner_nodes(self):
        mask = active_segments(self.xy, exclude_rear_face=True)
        np.testing.assert_array_equal(mask, [True, True, False, False, False])
        np.testing.assert_array_equal(active_nodes(mask), [True, True, True, False, False])
        np.testing.assert_array_equal(active_segments(self.xy), np.ones(5, dtype=bool))

    def test_direction_comes_from_geometry_not_freestream_or_global_x(self):
        rotated = np.column_stack([-self.xy[:, 1], self.xy[:, 0]])
        np.testing.assert_array_equal(active_segments(rotated, exclude_rear_face=True, nose_direction=[0, 4]),
                                      [True, True, False, False, False])
        with self.assertRaisesRegex(ThermalAnalysisError, "flat rear closure"):
            active_segments(self.xy, exclude_rear_face=True, nose_direction=[-1, 0])

    def test_ambiguous_rear_and_invalid_scope_are_rejected(self):
        concave = np.array([[0, 0], [0, 1], [.2, 1], [0, 2], [0, 3], [1, 1]])
        with self.assertRaisesRegex(ThermalAnalysisError, "single connected"):
            active_segments(concave, exclude_rear_face=True)
        for tolerance in (0, -.1, .02, float("nan")):
            with self.assertRaises(ThermalAnalysisError):
                active_segments(self.xy, exclude_rear_face=True, tolerance_fraction=tolerance)

    def test_corner_projection_never_averages_an_excluded_bad_anchor(self):
        mask = np.array([True, True, False, False, False])
        values = np.array([10., 20., -1e6, np.nan, -1e6])
        lengths = np.array([2., 1., 1., 1., 1.])
        np.testing.assert_allclose(scoped_face_values_to_nodes(values, lengths, mask), [10., 40/3, 20., 0., 0.])

    def test_migration_only_stales_calibration_and_thermal_and_preserves_explicit_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            old = runner_case(Path(directory) / "nose")
            old["calibration"].pop("exclude_rear_face", None)
            old["calibration"].pop("rear_plane_tolerance_fraction", None)
            new = single_solve_config(old)
            self.assertTrue(new["calibration"]["exclude_rear_face"])
            for key in ("geometry", "mesh", "flight", "physics", "thermal", "solver", "aero"):
                self.assertEqual(new[key], old[key])
            stages = {name: {"status": "complete"} for name in ("mesh", "solver", "temperature_export", "aero", "calibration", "thermal")}
            migrated = single_solve_state({"stages": stages}, old, new)["stages"]
            for stage in ("mesh", "solver", "temperature_export", "aero"):
                self.assertEqual(migrated[stage]["status"], "complete")
            for stage in ("calibration", "thermal"):
                self.assertEqual(migrated[stage]["status"], "stale")
            old["calibration"]["exclude_rear_face"] = False
            self.assertFalse(single_solve_config(old)["calibration"]["exclude_rear_face"])


class CalibrationReuseTests(unittest.TestCase):
    def save_cold_config(self, cfg):
        cold = calibration_cfg(cfg)
        path = calibration_paths(cfg)["cold_work"] / "_batch/config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cold))
        return cold, path

    def test_scope_only_change_reuses_both_verified_native_states(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = runner_case(Path(directory) / "nose")
            cfg["calibration"]["exclude_rear_face"] = False
            self.save_cold_config(cfg)
            paths = calibration_paths(cfg)
            original = {str(paths[f"{state}_work"] / name): file_sha256(paths[f"{state}_work"] / name)
                        for state in ("re", "cold") for name in ("04_final.cas.h5", "04_final.dat.h5")}
            cfg["calibration"]["exclude_rear_face"] = True
            with patch("pipeline.study.run_stage") as solve, patch("pipeline.cfd.export_saved_case") as export:
                result = run_calibration.run_calibration(cfg)
            self.assertEqual(result["status"], "complete")
            solve.assert_not_called(); export.assert_not_called()
            self.assertEqual(original, {path: file_sha256(path) for path in original})

    def test_changed_native_controls_or_stale_fields_cannot_reuse_cold_state(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = runner_case(Path(directory) / "nose")
            cold, path = self.save_cold_config(cfg)
            self.assertTrue(run_calibration.reusable_cold_export(cfg, cold, path))
            altered = copy.deepcopy(cold); altered["solver"]["courant_number"] = .15
            self.assertFalse(run_calibration.reusable_cold_export(cfg, altered, path))
            calibration_paths(cfg)["cold_faces"].write_text("changed export")
            self.assertFalse(run_calibration.reusable_cold_export(cfg, cold, path))

    def test_stage_archive_preserves_reusable_cold_pair_but_archives_old_thermal(self):
        from pipeline.study import _archive_stage
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "nose"
            cfg = runner_case(work)
            cfg["calibration"]["exclude_rear_face"] = False
            self.save_cold_config(cfg)
            cfg["calibration"]["exclude_rear_face"] = True
            thermal = work / "thermal_analysis"; thermal.mkdir()
            (thermal / "old_result.txt").write_text("previous full-boundary thermal field")
            reports = work / "reports"; reports.mkdir()
            (reports / "surface_heat_flux_history.png").write_bytes(b"old full-boundary plot")
            cold = calibration_paths(cfg)["cold_work"]
            original = {str(path.relative_to(cold)): path.read_bytes() for path in cold.rglob("*") if path.is_file()}
            _archive_stage("calibration", work, cfg, {"stages": {}})
            self.assertEqual(original, {str(path.relative_to(cold)): path.read_bytes() for path in cold.rglob("*") if path.is_file()})
            self.assertFalse(thermal.exists())
            self.assertFalse(reports.exists())
            self.assertEqual(len(list((work / "_history").glob("*/thermal_analysis/old_result.txt"))), 1)
            self.assertEqual(len(list((work / "_history").glob("*/reports/surface_heat_flux_history.png"))), 1)


if __name__ == "__main__":unittest.main()
