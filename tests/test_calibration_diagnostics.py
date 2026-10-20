"""Offline face diagnostics preserve signed anchors and calibration safeguards."""
import builtins
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import aerothermal_calibration as calibration
from pipeline.reporting import read_csv, write_csv
from pipeline.thermal_loading import ThermalAnalysisError
from test_aerothermal_calibration import calibration_case, refresh_export


def snapshot_files(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def make_negative_face(cfg, index=1):
    paths = calibration.calibration_paths(cfg)
    re_faces, cold_faces = read_csv(paths["re_faces"]), read_csv(paths["cold_faces"])
    teq = float(re_faces[index]["wall_temperature_K"])
    thermal = cfg["thermal"]
    qeq = thermal["emissivity"] * calibration.STEFAN_BOLTZMANN * (
        teq**4 - thermal["external_radiation_temperature_k"]**4)
    cold_faces[index]["heat_flux_into_wall_W_m2"] = .5 * qeq
    cold_faces[index]["raw_heat_flux_W_m2"] = -.5 * qeq
    write_csv(paths["cold_faces"], cold_faces)
    refresh_export(cfg, "cold")
    return -.5 * qeq / (teq - cfg["calibration"]["cold_wall_temperature_k"])


class CalibrationDiagnosticsTests(unittest.TestCase):
    def test_passed_diagnosis_is_read_only_and_does_not_import_ansys_engines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = calibration_case(root / "nose")
            before = snapshot_files(root)
            native_import = builtins.__import__
            def without_engine(name, *args, **kwargs):
                if name.startswith(("ansys.fluent", "ansys.mapdl", "ansys.geometry")):
                    raise AssertionError("Calibration diagnosis must not import an Ansys client")
                return native_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=without_engine):
                result = calibration.diagnose_calibration(cfg)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["face_count"], 4)
            self.assertEqual(result["invalid_face_count"], 0)
            self.assertEqual(result["failed_boundary_length_fraction"], 0.)
            self.assertAlmostEqual(result["minimum_conductance_W_m2K"], 100.)
            self.assertLessEqual(len(result["worst_faces"]), 10)
            json.dumps(result, allow_nan=False)
            self.assertEqual(snapshot_files(root), before)
            self.assertNotIn("diagnostics_csv", result)

    def test_negative_conductance_is_identified_despite_passed_cold_cfd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = calibration_case(root / "nose")
            expected = make_negative_face(cfg)
            paths = calibration.calibration_paths(cfg)
            self.assertEqual(calibration.verify_calibration_export(cfg, "cold")["status"], "complete")
            before = snapshot_files(root)
            result = calibration.diagnose_calibration(cfg, write=False)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["face_count"], 4)
            self.assertEqual(result["invalid_face_count"], 1)
            self.assertAlmostEqual(result["failed_boundary_length_fraction"], .25)
            self.assertAlmostEqual(result["minimum_conductance_W_m2K"], expected)
            bad = next(row for row in result["worst_faces"] if row["status"] == "nonpositive_conductance")
            self.assertEqual(bad["surface_order_index"], 1)
            self.assertEqual(bad["zone_name"], "body_wall")
            self.assertEqual((bad["re_face_index"], bad["cold_face_index"]), (1, 1))
            self.assertEqual((bad["x_m"], bad["y_m"]), (1., .5))
            self.assertAlmostEqual(bad["candidate_conductance_W_m2K"], expected)
            self.assertGreater(bad["cold_heat_flux_into_wall_W_m2"], 0.)
            self.assertLess(bad["cold_heat_flux_into_wall_W_m2"], bad["equilibrium_convective_heat_flux_W_m2"])
            self.assertEqual(snapshot_files(root), before)
            self.assertFalse(paths["summary"].exists())
            json.dumps(result, allow_nan=False)

    def test_face_matching_uses_geometry_when_native_ids_and_loop_order_differ(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            expected = make_negative_face(cfg)
            paths = calibration.calibration_paths(cfg)
            for kind in ("faces", "nodes"):
                rows = read_csv(paths[f"cold_{kind}"])
                rows = rows[1:] + rows[:1]
                for index, row in enumerate(rows):
                    row["surface_order_index"] = index
                    if kind == "faces":row["face_index"] = 1000 + index
                write_csv(paths[f"cold_{kind}"], rows)
            refresh_export(cfg, "cold")
            result = calibration.diagnose_calibration(cfg)
            self.assertEqual(result["invalid_face_count"], 1)
            bad = next(row for row in result["worst_faces"] if row["status"] == "nonpositive_conductance")
            self.assertEqual((bad["re_face_index"], bad["cold_face_index"]), (1, 1000))
            self.assertEqual((bad["x_m"], bad["y_m"]), (1., .5))
            self.assertAlmostEqual(bad["candidate_conductance_W_m2K"], expected)

    def test_failed_boundary_fraction_weights_edge_lengths_not_face_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            make_negative_face(cfg)  # failing right-hand edge becomes length 2
            paths = calibration.calibration_paths(cfg)
            for state in ("re", "cold"):
                nodes = read_csv(paths[f"{state}_nodes"])
                for row in nodes:row["y_m"] = 2 * float(row["y_m"])
                write_csv(paths[f"{state}_nodes"], nodes)
                faces = read_csv(paths[f"{state}_faces"])
                for row in faces:
                    for name in ("y_m", "start_y_m", "end_y_m"):row[name] = 2 * float(row[name])
                    if float(row["start_x_m"]) == float(row["end_x_m"]):row["segment_length_m"] = 2.
                write_csv(paths[f"{state}_faces"], faces)
                refresh_export(cfg, state)
            result = calibration.diagnose_calibration(cfg)
            self.assertEqual(result["invalid_face_count"], 1)
            self.assertEqual(result["face_count"], 4)
            self.assertAlmostEqual(result["failed_boundary_length_fraction"], 2 / 6)

    def test_failed_build_writes_diagnostics_but_never_a_usable_load_package(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            make_negative_face(cfg)
            paths = calibration.calibration_paths(cfg)
            result = calibration.diagnose_calibration(cfg, write=True)
            csv_path, json_path = Path(result["diagnostics_csv"]), Path(result["diagnostics_json"])
            self.assertEqual(csv_path, paths["root"] / "thermal_bc/calibration_diagnostics.csv")
            self.assertEqual(json_path, paths["root"] / "thermal_bc/calibration_diagnostics.json")
            rows = read_csv(csv_path)
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(row["status"] != "passed" for row in rows), 1)
            self.assertEqual(json.loads(json_path.read_text())["invalid_face_count"], 1)
            with self.assertRaises(ThermalAnalysisError) as caught:
                calibration.build_surface_package(cfg)
            message = str(caught.exception)
            self.assertIn("1", message)
            self.assertIn(str(csv_path), message)
            self.assertIn(str(json_path), message)
            self.assertFalse(paths["summary"].exists())
            self.assertFalse(paths["nodes"].exists())
            self.assertFalse(paths["faces"].exists())

    def test_insufficient_temperature_separation_has_distinct_face_status(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            paths = calibration.calibration_paths(cfg)
            for kind in ("faces", "nodes"):
                rows = read_csv(paths[f"re_{kind}"])
                rows[2]["wall_temperature_K"] = 310.
                write_csv(paths[f"re_{kind}"], rows)
            refresh_export(cfg, "re")
            result = calibration.diagnose_calibration(cfg)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["invalid_face_count"], 1)
            bad = next(row for row in result["worst_faces"] if row["status"] == "insufficient_temperature_separation")
            self.assertEqual(bad["surface_order_index"], 2)
            self.assertEqual(bad["equilibrium_temperature_K"], 310.)
            self.assertEqual(bad["cold_wall_temperature_K"], 300.)
            with self.assertRaises(ThermalAnalysisError):calibration.build_surface_package(cfg)

    def test_zero_temperature_separation_is_reported_before_division_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            paths = calibration.calibration_paths(cfg)
            for kind in ("faces", "nodes"):
                rows = read_csv(paths[f"re_{kind}"])
                rows[2]["wall_temperature_K"] = 300.
                write_csv(paths[f"re_{kind}"], rows)
            refresh_export(cfg, "re")
            result = calibration.diagnose_calibration(cfg)
            self.assertEqual(result["invalid_face_count"], 1)
            bad = next(row for row in result["worst_faces"] if row["surface_order_index"] == 2)
            self.assertEqual(bad["status"], "insufficient_temperature_separation")
            self.assertIsNone(bad["candidate_conductance_W_m2K"])
            json.dumps(result, allow_nan=False)
            with self.assertRaises(ThermalAnalysisError):calibration.build_surface_package(cfg)

    def test_old_successful_package_is_preserved_but_rejected_after_failed_recalibration(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            paths = calibration.calibration_paths(cfg)
            calibration.build_surface_package(cfg)
            self.assertEqual(calibration.verify_surface_package(cfg)[0], paths["nodes"])
            original = {name: paths[name].read_bytes() for name in ("summary", "faces", "nodes")}
            make_negative_face(cfg)
            with self.assertRaises(ThermalAnalysisError):calibration.build_surface_package(cfg)
            self.assertEqual({name: paths[name].read_bytes() for name in original}, original)
            with self.assertRaisesRegex(ThermalAnalysisError, "changed after"):
                calibration.verify_surface_package(cfg)
            diagnostics = paths["root"] / "thermal_bc/calibration_diagnostics.json"
            self.assertEqual(json.loads(diagnostics.read_text())["status"], "failed")

    def test_cli_emits_json_status_exit_code_and_never_changes_saved_files(self):
        import study
        for state in ("passed", "failed", "unavailable"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cfg = calibration_case(root / "nose")
                if state == "failed":make_negative_face(cfg)
                elif state == "unavailable":
                    paths = calibration.calibration_paths(cfg)
                    (paths["cold_work"] / "04_final.dat.h5").write_bytes(b"changed checkpoint")
                before, output = snapshot_files(root), io.StringIO()
                native_import = builtins.__import__
                def without_engine(name, *args, **kwargs):
                    if name.startswith(("ansys.fluent", "ansys.mapdl", "ansys.geometry")):
                        raise AssertionError("Read-only status must not launch Ansys")
                    return native_import(name, *args, **kwargs)
                with patch("pipeline.config.load_config", return_value=cfg) as load, \
                     patch("sys.argv", ["study.py", "calibration-status", "--config", "selected-case.yaml"]), \
                     patch("builtins.__import__", side_effect=without_engine), redirect_stdout(output):
                    code = study.main()
                load.assert_called_once_with("selected-case.yaml")
                result = json.loads(output.getvalue())
                self.assertEqual(result["status"], state)
                self.assertEqual(code, 0 if state == "passed" else 2)
                if state == "unavailable":self.assertIn("error", result)
                self.assertEqual(snapshot_files(root), before)

    def test_overflowing_radiation_anchor_remains_json_safe_and_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = calibration_case(Path(directory) / "nose")
            paths = calibration.calibration_paths(cfg)
            # The exported number is finite, but its fourth power is not.
            for kind in ("faces", "nodes"):
                rows = read_csv(paths[f"re_{kind}"])
                rows[2]["wall_temperature_K"] = 1e100
                write_csv(paths[f"re_{kind}"], rows)
            refresh_export(cfg, "re")
            result = calibration.diagnose_calibration(cfg)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["invalid_face_count"], 1)
            bad = next(row for row in result["worst_faces"] if row["status"] == "nonfinite_calibration")
            self.assertEqual(bad["surface_order_index"], 2)
            json.dumps(result, allow_nan=False)

    def test_provenance_and_geometric_mismatches_cannot_be_diagnosed_as_valid(self):
        for changed in ("cold_data", "geometry"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                cfg = calibration_case(Path(directory) / "nose")
                paths = calibration.calibration_paths(cfg)
                if changed == "cold_data":
                    (paths["cold_work"] / "04_final.dat.h5").write_bytes(b"unrelated CFD checkpoint")
                else:
                    rows = read_csv(paths["cold_faces"])
                    rows[0]["x_m"] = .55
                    write_csv(paths["cold_faces"], rows)
                    refresh_export(cfg, "cold")
                with self.assertRaises(ThermalAnalysisError):calibration.diagnose_calibration(cfg)


if __name__ == "__main__":unittest.main()
