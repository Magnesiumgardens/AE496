"""Signed endpoint calibration, matched sources, and spatial Mechanical loads."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from pipeline.convection_loading import ConvectionLoad, read_load, validate_settings
from pipeline.endpoint_loading import (build_convection_package, convection_paths, derive_coefficients,
                                       face_values_to_nodes, verify_convection_package)
from pipeline.reporting import read_csv, write_csv
from pipeline.thermal import apply_surface_loads, run_thermal_analysis
from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.utils import atomic_json, file_sha256
from test_endpoint_reports import endpoint_case

ROOT = Path(__file__).resolve().parents[1]


def refresh_endpoint_manifest(work, kind):
    endpoint = work / 'cfd' / kind
    path = endpoint / 'thermal_bc/endpoint_summary.json'
    metadata = json.loads(path.read_text())
    metadata['nodes_csv_sha256'] = file_sha256(endpoint / 'thermal_bc/wall_nodes.csv')
    metadata['faces_csv_sha256'] = file_sha256(endpoint / 'thermal_bc/wall_faces.csv')
    atomic_json(path, metadata)


class EndpointCalibrationTests(unittest.TestCase):
    def test_signed_coefficients_and_small_temperature_separation(self):
        cold = np.array([[0., 0., 0., 300., 17000.], [1., 0., 0., 300., -2000.]])
        recovery = np.array([[0., 0., 0., 2000.], [1., 0., 0., 200.]])
        np.testing.assert_allclose(derive_coefficients(cold, recovery), [10, 20])
        wrong_sign = cold.copy(); wrong_sign[0, 4] *= -1
        with self.assertRaisesRegex(ThermalAnalysisError, 'sign is inconsistent'):
            derive_coefficients(wrong_sign, recovery)
        close = recovery.copy(); close[0, 3] = 310
        with self.assertRaisesRegex(ThermalAnalysisError, 'Cannot identify h'):
            derive_coefficients(cold, close)
        shifted = recovery.copy(); shifted[1, 0] += 1e-3
        with self.assertRaisesRegex(ThermalAnalysisError, 'vertices/order do not match'):
            derive_coefficients(cold, shifted)
        cold[0, 4] = 0
        self.assertEqual(derive_coefficients(cold, recovery)[0], 0)

    def test_face_collocation_precedes_length_weighted_nodal_projection(self):
        np.testing.assert_allclose(face_values_to_nodes([10, 20, 30], [1, 2, 3]), [25, 50 / 3, 26])
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'nose'; cfg = endpoint_case(work)
            # Deliberately change interpolated nodal temperatures. Calibration
            # must retain its collocated face temperatures rather than divide
            # projected flux by independently interpolated boundary values.
            nodes_path = work / 'cfd/adiabatic/thermal_bc/wall_nodes.csv'
            rows = read_csv(nodes_path)
            for row in rows:row['wall_temperature_K'] = 3500
            write_csv(nodes_path, rows); refresh_endpoint_manifest(work, 'adiabatic')
            build_convection_package(cfg)
            rows = read_csv(convection_paths(cfg)[0])
            face_h = np.array([17000, 19000, 21000, 23000]) / (np.array([2000, 2100, 2200, 2300]) - 300)
            np.testing.assert_allclose([float(row['heat_transfer_coefficient_W_m2K']) for row in rows],
                                       face_values_to_nodes(face_h, np.ones(4)))
            np.testing.assert_allclose([float(row['recovery_temperature_K']) for row in rows],
                                       [2150, 2050, 2150, 2250])
            verify_convection_package(cfg)

    def test_face_geometry_and_prescribed_temperature_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'nose'; cfg = endpoint_case(work)
            path = work / 'cfd/adiabatic/thermal_bc/wall_faces.csv'
            faces = read_csv(path); faces[0]['start_x_m'] = 0.1
            write_csv(path, faces); refresh_endpoint_manifest(work, 'adiabatic')
            with self.assertRaisesRegex(ThermalAnalysisError, 'zones or geometry differ'):
                build_convection_package(cfg)
            faces[0]['start_x_m'] = 0.0
            write_csv(path, faces); refresh_endpoint_manifest(work, 'adiabatic')
            path = work / 'cfd/cold_wall/thermal_bc/wall_nodes.csv'
            nodes = read_csv(path); nodes[0]['wall_temperature_K'] = 350
            write_csv(path, nodes); refresh_endpoint_manifest(work, 'cold_wall')
            with self.assertRaisesRegex(ThermalAnalysisError, 'prescribed wall temperature'):
                build_convection_package(cfg)

    def test_parallel_local_face_indices_do_not_change_physical_sample_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)/'nose'; cfg = endpoint_case(work)
            old = read_csv(convection_paths(cfg)[0])
            path = work/'cfd/adiabatic/thermal_bc/wall_faces.csv'
            rows = read_csv(path)
            for index,row in enumerate(rows):row['face_index'] = 100-index
            write_csv(path,rows); refresh_endpoint_manifest(work,'adiabatic')
            build_convection_package(cfg)
            new = read_csv(convection_paths(cfg)[0])
            self.assertEqual(old,new)

    def test_source_and_mesh_changes_cannot_reuse_a_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'nose'; cfg = endpoint_case(work)
            verify_convection_package(cfg)
            cfg['workflow']['min_delta_temperature_k'] = 30
            with self.assertRaisesRegex(ThermalAnalysisError, 'derivation settings changed'):
                verify_convection_package(cfg)
            cfg['workflow']['min_delta_temperature_k'] = 25
            (work / '02_mesh.msh.h5').write_bytes(b'new mesh fixture')
            with self.assertRaisesRegex(ThermalAnalysisError, 'mesh differs'):
                verify_convection_package(cfg)

    def test_identical_case_files_cannot_represent_distinct_wall_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'nose'; cfg = endpoint_case(work)
            shutil.copyfile(work / 'cfd/cold_wall/04_final.cas.h5', work / 'cfd/adiabatic/04_final.cas.h5')
            for action in (build_convection_package, verify_convection_package):
                with self.assertRaisesRegex(ThermalAnalysisError, 'identical case files'):
                    action(cfg)

    def test_updated_cold_checkpoint_requires_a_new_warm_adiabatic_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'nose'; cfg = endpoint_case(work)
            (work / 'cfd/cold_wall/04_final.dat.h5').write_bytes(b'new converged cold checkpoint')
            summary_path = work / 'cfd/cold_wall/summary.json'
            summary = json.loads(summary_path.read_text())
            source = {key: file_sha256(work / 'cfd/cold_wall' / name) for key, name in
                      (('final_case', '04_final.cas.h5'), ('final_data', '04_final.dat.h5'))}
            summary['numerical_quality']['source_sha256'] = source
            atomic_json(summary_path, summary)
            metadata_path = work / 'cfd/cold_wall/thermal_bc/endpoint_summary.json'
            metadata = json.loads(metadata_path.read_text()); metadata['source_sha256'] = source
            atomic_json(metadata_path, metadata)
            with self.assertRaisesRegex(ThermalAnalysisError, 'not warm-started from the current cold-wall'):
                build_convection_package(cfg)

    def test_a_changed_reference_wall_temperature_cannot_use_old_endpoint_data(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'nose'; cfg = endpoint_case(work)
            cfg['thermal']['wall_temperature_k'] = 400
            with self.assertRaisesRegex(ThermalAnalysisError, 'temperature differs from the selected'):
                build_convection_package(cfg)

    def test_field_and_flux_sign_changes_cannot_relabel_old_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'nose'; cfg = endpoint_case(work)
            for key, replacement in (('heat_flux_into_wall_sign', 1), ('heat_flux_field_name', 'other-heat-flux'),
                                     ('temperature_field_name', 'other-temperature')):
                with self.subTest(setting=key):
                    original = cfg['thermal_export'][key]; cfg['thermal_export'][key] = replacement
                    for action in (build_convection_package, verify_convection_package):
                        with self.assertRaisesRegex(ThermalAnalysisError, 'field or heat-flux sign settings changed'):
                            action(cfg)
                    cfg['thermal_export'][key] = original

    def test_changed_cad_cannot_feed_unchanged_mesh_calibration_to_mechanical(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'nose'; cfg = endpoint_case(work)
            Path(cfg['geometry']['input_file']).write_bytes(b'a different actual CAD shape')
            for action in (build_convection_package, verify_convection_package):
                with self.assertRaisesRegex(ThermalAnalysisError, 'CAD source differs'):
                    action(cfg)


class SpatialMechanicalTests(unittest.TestCase):
    def test_mapdl_receives_local_coefficients_and_bulk_temperatures_at_both_line_ends(self):
        load = ConvectionLoad(np.array([[0., 0.], [1., 0.], [0., 1.]]),
                              np.array([2000., 2200., 2400.]), np.array([10., 20., 30.]))
        calls = []
        engine = SimpleNamespace(sfl=lambda *args: calls.append(args))
        apply_surface_loads(engine, [11, 12, 13], load,
                            dict(include_rad=True, emissivity=.9, trad=226.5))
        convection = [call for call in calls if call[1] == 'CONV']
        self.assertEqual(convection, [(11, 'CONV', 10., 20., 2000., 2200.),
                                      (12, 'CONV', 20., 30., 2200., 2400.),
                                      (13, 'CONV', 30., 10., 2400., 2000.)])
        self.assertFalse(any(call[1] == 'RAD' for call in calls))  # Added later as SURF151.
        np.testing.assert_allclose(load.incoming_flux([300, 1000, 2000]), [17000, 24000, 12000])

    def test_spatial_coefficients_remain_attached_when_polygon_orientation_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nodes.csv'
            write_csv(path, [dict(surface_order_index=i, x_m=x, y_m=y,
                                 recovery_temperature_K=2000 + i, heat_transfer_coefficient_W_m2K=10 + i)
                             for i, (x, y) in enumerate(((0., 0.), (0., 1.), (1., 0.)))])
            load = read_load(path)
            np.testing.assert_allclose(load.xy, [[1, 0], [0, 1], [0, 0]])
            np.testing.assert_allclose(load.coefficient, [12, 11, 10])
            np.testing.assert_allclose(load.gas_temperature, [2002, 2001, 2000])

    def test_endpoint_thermal_runner_saves_spatial_package_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); work = root / 'nose'; cfg = endpoint_case(work)
            thermal = yaml.safe_load((ROOT / 'tests/fixtures/endpoint_thermal_analysis.yaml').read_text())
            thermal['loading']['mode'] = 'convective_from_cfd'
            thermal['transient'].update(duration_s=2, initial_time_step_s=1, min_time_step_s=.1, max_time_step_s=2)
            thermal['visualization']['enabled'] = False
            thermal_path = root / 'thermal.yaml'; thermal_path.write_text(yaml.safe_dump(thermal))
            settings = validate_settings(cfg, thermal)
            self.assertEqual(settings['h_source'], 'two_endpoint')
            self.assertNotIn('h', settings)
            engine = SimpleNamespace(exit=lambda: None)
            def solve(mapdl, load, received, size):
                self.assertIsInstance(load.coefficient, np.ndarray)
                self.assertGreater(np.ptp(load.coefficient), 0)
                return np.arange(1, 5), np.c_[load.xy, np.zeros(4)], [np.array([400., 500., 600., 700.]),
                    np.array([500., 600., 700., 800.])], [1., 2.], 2
            with patch('ansys.mapdl.core.launch_mapdl', return_value=engine), \
                 patch('pipeline.thermal.solve_thermal', side_effect=solve):
                result = run_thermal_analysis(cfd_config=work / '_batch/config.yaml', thermal_config=thermal_path)
            self.assertEqual(result['model_version'], '4.0-endpoint-convection')
            self.assertEqual(result['loading_mode'], 'convective_from_cfd')
            self.assertEqual(set(result['source_sha256']), {'cold_case', 'cold_data', 'adiabatic_case', 'adiabatic_data'})
            self.assertEqual(result['boundary_csv_sha256'], file_sha256(convection_paths(cfg)[0]))
            self.assertEqual(result['convection_package_sha256'], file_sha256(convection_paths(cfg)[1]))
            self.assertEqual(result['analysis']['duration_s'], 2)
            self.assertEqual(len(read_csv(work / 'thermal_analysis/wall_temperature_history.csv')), 12)


if __name__ == '__main__':unittest.main()
