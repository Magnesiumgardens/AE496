"""Rear exclusion is geometric and cannot bypass invalid forward heating."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.aerothermal_calibration import (
    STEFAN_BOLTZMANN, build_surface_package, calibration_paths, calibration_settings,
    diagnose_calibration, verify_calibration_export, verify_surface_package,
)
from pipeline.reporting import read_csv, write_csv
from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.utils import atomic_json, file_sha256
from test_aerothermal_calibration import calibration_case, refresh_export


def split_rear_case(work):
    """Rectangle with two collinear rear faces and an interior rear vertex."""
    cfg = calibration_case(work)
    cfg["calibration"]["exclude_rear_face"] = True
    cfg["reporting"] = dict(nose_direction=[1., 0.])
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., .5]])
    temperatures = np.array([1200., 1300., 1400., 1500., 1500.])
    qeq = .9 * STEFAN_BOLTZMANN * (temperatures**4 - 3.**4)
    qcold = qeq + np.array([100., 120., 140., 160., 180.]) * (temperatures - 300.)
    qcold[3:] = qeq[3:] / 2  # both rear anchors fail the positive-slope fit
    paths = calibration_paths(cfg)
    for state in ("re", "cold"):
        faces, nodes = [], []
        for index, point in enumerate(points):
            end = points[(index + 1) % len(points)]
            t = 300. if state == "cold" else float(temperatures[index])
            nodes.append(dict(surface_order_index=index, x_m=point[0], y_m=point[1], z_m=0., wall_temperature_K=t))
            face = dict(surface_order_index=index, zone_name="body_wall", face_index=100+index,
                x_m=(point[0]+end[0])/2, y_m=(point[1]+end[1])/2, z_m=0.,
                segment_length_m=float(np.linalg.norm(end-point)), start_x_m=point[0], start_y_m=point[1],
                end_x_m=end[0], end_y_m=end[1], wall_temperature_K=t)
            if state == "cold":face.update(heat_flux_into_wall_W_m2=qcold[index], raw_heat_flux_W_m2=-qcold[index])
            faces.append(face)
        write_csv(paths[f"{state}_faces"], faces)
        write_csv(paths[f"{state}_nodes"], nodes)
        refresh_export(cfg, state)
    return cfg


class RearCalibrationTests(unittest.TestCase):
    def test_bad_rear_anchors_are_retained_as_diagnostics_but_not_calibrated(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = split_rear_case(Path(directory)/"nose")
            diagnosis = diagnose_calibration(cfg)
            self.assertEqual(diagnosis["status"], "passed")
            self.assertEqual(diagnosis["invalid_face_count"], 0)
            self.assertEqual(diagnosis["excluded_face_count"], 2)
            self.assertEqual(diagnosis["excluded_invalid_face_count"], 2)
            self.assertTrue(all(row["candidate_conductance_W_m2K"] < 0 for row in diagnosis["excluded_faces"]))
            metadata = build_surface_package(cfg)
            paths = calibration_paths(cfg)
            faces, nodes = read_csv(paths["faces"]), read_csv(paths["nodes"])
            self.assertEqual(len(faces), 5)
            self.assertEqual(len(nodes), 5)
            self.assertEqual([int(row["thermal_active"]) for row in faces], [1,1,1,0,0])
            self.assertEqual([int(row["thermal_active"]) for row in nodes], [1,1,1,1,0])
            self.assertEqual([int(row["thermal_segment_active"]) for row in nodes], [1,1,1,0,0])
            self.assertEqual(metadata["thermal_scope"]["rear_boundary"], "insulated")
            self.assertEqual(verify_surface_package(cfg)[0], paths["nodes"])

    def test_projection_uses_only_active_neighbors_at_rear_corners(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = split_rear_case(Path(directory)/"nose")
            metadata = build_surface_package(cfg)
            paths = calibration_paths(cfg)
            nodes = read_csv(paths["nodes"])
            np.testing.assert_allclose([float(row["aerodynamic_conductance_W_m2K"]) for row in nodes], [100.,110.,130.,140.,0.])
            self.assertEqual(float(nodes[-1]["driving_temperature_K"]), 0.)
            raw_faces = read_csv(paths["cold_faces"])
            expected = sum(float(row["segment_length_m"])*float(row["heat_flux_into_wall_W_m2"]) for row in raw_faces[:3])
            self.assertAlmostEqual(metadata["diagnostics"]["cold_face_heat_input_per_unit_span_W_m"], expected)
            before = copy.deepcopy(nodes)
            # Even a large change on excluded faces cannot leak into corner H/B.
            for row in raw_faces[3:]:
                row["heat_flux_into_wall_W_m2"] = -1e8
                row["raw_heat_flux_W_m2"] = 1e8
            write_csv(paths["cold_faces"], raw_faces)
            refresh_export(cfg, "cold")
            build_surface_package(cfg)
            self.assertEqual(read_csv(paths["nodes"]), before)

    def test_invalid_forward_face_is_not_excluded_to_make_calibration_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = split_rear_case(Path(directory)/"nose")
            paths = calibration_paths(cfg)
            rows = read_csv(paths["cold_faces"])
            rows[1]["heat_flux_into_wall_W_m2"] = 1.
            rows[1]["raw_heat_flux_W_m2"] = -1.
            write_csv(paths["cold_faces"], rows)
            refresh_export(cfg, "cold")
            diagnosis = diagnose_calibration(cfg)
            self.assertEqual(diagnosis["status"], "failed")
            self.assertEqual(diagnosis["invalid_face_count"], 1)
            self.assertEqual(diagnosis["worst_faces"][0]["surface_order_index"], 1)
            with self.assertRaisesRegex(ThermalAnalysisError, "positive aerodynamic conductance"):
                build_surface_package(cfg)
            self.assertFalse(paths["summary"].exists())

    def test_scope_changes_do_not_invalidate_native_exports_but_invalidate_package(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = split_rear_case(Path(directory)/"nose")
            build_surface_package(cfg)
            for state in ("re", "cold"):
                self.assertEqual(verify_calibration_export(cfg,state)["status"], "complete")
            cfg["calibration"]["exclude_rear_face"] = False
            with self.assertRaisesRegex(ThermalAnalysisError, "stale"):
                verify_surface_package(cfg)
            cfg["calibration"]["exclude_rear_face"] = True
            cfg["reporting"]["nose_direction"] = [0., 1.]
            with self.assertRaisesRegex(ThermalAnalysisError, "stale"):
                verify_surface_package(cfg)

    def test_package_flags_are_verified_against_geometric_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = split_rear_case(Path(directory)/"nose")
            build_surface_package(cfg)
            paths = calibration_paths(cfg)
            rows = read_csv(paths["faces"])
            rows[-1]["thermal_active"] = 1
            write_csv(paths["faces"], rows)
            metadata = json.loads(paths["summary"].read_text())
            metadata["faces_csv_sha256"] = file_sha256(paths["faces"])
            atomic_json(paths["summary"], metadata)
            with self.assertRaisesRegex(ThermalAnalysisError, "scope flags"):
                verify_surface_package(cfg)

    def test_legacy_default_keeps_all_faces_and_requires_positive_fit_everywhere(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = split_rear_case(Path(directory)/"nose")
            cfg["calibration"].pop("exclude_rear_face")
            self.assertFalse(calibration_settings(cfg)["exclude_rear_face"])
            with self.assertRaisesRegex(ThermalAnalysisError, "positive aerodynamic conductance"):
                build_surface_package(cfg)


if __name__ == "__main__":unittest.main()
