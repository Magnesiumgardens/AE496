"""Endpoint provenance and recovery/solid report separation, without Ansys."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from pipeline.aero import aerodynamic_summary
from pipeline.endpoint_loading import build_convection_package, convection_paths, source_hashes
from pipeline.gas_reporting import (aero_complete_for_report, build_gas_row, create_gas_reports,
                                    report_cfg, source_cfg, source_work, temperature_complete)
from pipeline.mechanical_reporting import mechanical_complete, create_mechanical_reports
from pipeline.reporting import write_csv
from pipeline.temperature_export import export_temperatures, temperature_conditions, temperature_paths
from pipeline.utils import atomic_json, file_sha256
from test_adiabatic_study import config, field_solver
from test_nose_trade import fluent


def endpoint_case(work, flux_scale=1):
    cfg = config(work)
    cfg['thermal_export'] = dict(enabled=True, temperature_field_name='temperature',
                                heat_flux_field_name='heat-flux', heat_flux_into_wall_sign=-1)
    cfg['workflow'] = dict(strategy='two_endpoint', min_delta_temperature_k=25,
                           coordinate_tolerance_m=1e-9)
    cfg['thermal'] = dict(mode='fixed_temperature', wall_temperature_k=300.0)
    cfg['thermal_export'] = dict(temperature_field_name='temperature', heat_flux_field_name='heat-flux',
                                 heat_flux_into_wall_sign=-1)
    (work / '_batch').mkdir(parents=True)
    Path(cfg['geometry']['input_file']).write_bytes(b'shared CAD fixture')
    (work / '_batch/config.yaml').write_text(yaml.safe_dump(cfg))
    (work / '02_mesh.msh.h5').write_bytes(b'shared mesh fixture')
    xy = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    for kind in ('cold_wall', 'adiabatic'):
        endpoint = work / 'cfd' / kind
        (endpoint / 'thermal_bc').mkdir(parents=True)
        for name in ('04_final.cas.h5', '04_final.dat.h5'):
            (endpoint / name).write_bytes((work.name + kind + name).encode())
        source = {key: file_sha256(endpoint / name) for key, name in
                  (('final_case', '04_final.cas.h5'), ('final_data', '04_final.dat.h5'))}
        solver_summary = dict(thermal_mode='adiabatic' if kind == 'adiabatic' else 'fixed_temperature',
            numerical_quality=dict(status='residual_screen_passed', source_sha256=source),
            source_geometry_sha256=file_sha256(cfg['geometry']['input_file']))
        if kind == 'adiabatic':
            solver_summary['reference_source_sha256'] = {key: file_sha256(work / 'cfd/cold_wall' / name)
                for key, name in (('final_case', '04_final.cas.h5'), ('final_data', '04_final.dat.h5'))}
        atomic_json(endpoint / 'summary.json', solver_summary)
        nodes, faces = [], []
        for index, point in enumerate(xy):
            nxt = xy[(index + 1) % len(xy)]
            temperature = 300.0 if kind == 'cold_wall' else 2000.0 + 100 * index
            row = dict(surface_order_index=index, x_m=float(point[0]), y_m=float(point[1]), z_m=0.,
                       wall_temperature_K=temperature)
            face = dict(row, x_m=float((point[0] + nxt[0]) / 2), y_m=float((point[1] + nxt[1]) / 2),
                        zone_name='body_wall', face_index=index, segment_length_m=1.,
                        start_x_m=float(point[0]), start_y_m=float(point[1]), end_x_m=float(nxt[0]), end_y_m=float(nxt[1]))
            if kind == 'cold_wall':
                row['heat_flux_into_wall_W_m2'] = flux_scale * (17000 + 2000 * index)
                face['heat_flux_into_wall_W_m2'] = row['heat_flux_into_wall_W_m2']
            nodes.append(row); faces.append(face)
        node_path = write_csv(endpoint / 'thermal_bc/wall_nodes.csv', nodes)
        face_path = write_csv(endpoint / 'thermal_bc/wall_faces.csv', faces)
        conditions = temperature_conditions(cfg)
        if kind == 'adiabatic':conditions['thermal'] = {'mode': 'adiabatic'}
        atomic_json(endpoint / 'thermal_bc/endpoint_summary.json', dict(status='complete', conditions=conditions,
            source_sha256=source, nodes_csv_sha256=file_sha256(node_path), faces_csv_sha256=file_sha256(face_path),
            mesh_sha256=file_sha256(work / '02_mesh.msh.h5'), fields={
                'temperature':cfg['thermal_export']['temperature_field_name'], **({
                    'heat_flux':cfg['thermal_export']['heat_flux_field_name'],
                    'heat_flux_into_wall_sign':cfg['thermal_export']['heat_flux_into_wall_sign']}
                    if kind == 'cold_wall' else {})}))
    gas_cfg = report_cfg(cfg)
    metadata = export_temperatures(field_solver()[0], gas_cfg, ['body_wall'])
    source = {key: file_sha256(source_work(cfg) / name) for key, name in
              (('final_case', '04_final.cas.h5'), ('final_data', '04_final.dat.h5'))}
    metadata.update(source_sha256=source, wall_csv_sha256=file_sha256(temperature_paths(gas_cfg)[0]))
    atomic_json(temperature_paths(gas_cfg)[1], metadata)
    aero = aerodynamic_summary(fluent(), source_cfg(cfg), ['body_wall.1', 'body_wall.2'])
    aero['source_sha256'] = source
    atomic_json(work / 'aero_summary.json', aero)
    build_convection_package(cfg)
    return cfg


def solid_result(work, cfg, thermal_path):
    output = work / 'thermal_analysis'; output.mkdir()
    xy = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    wall = [dict(time_s=time, surface_order_index=i, x_m=x, y_m=y, wall_temperature_K=300 + time + i)
            for time in (0, 200) for i, (x, y) in enumerate(xy)]
    write_csv(output / 'wall_temperature_history.csv', wall)
    write_csv(output / 'temperature_history.csv', [dict(time_s=200, minimum_K=500, mean_K=501.5, maximum_K=503)])
    write_csv(output / 'thermal_nodes_final.csv', [dict(node_id=i + 1, x_m=x, y_m=y, temperature_K=500 + i)
                                                 for i, (x, y) in enumerate(xy)])
    nodes, package_path = convection_paths(cfg)
    package = json.loads(package_path.read_text())
    atomic_json(output / 'thermal_summary.json', dict(status='complete', model_version='4.0-endpoint-convection',
        loading_mode='convective_from_cfd', thermal_config_path=str(thermal_path),
        thermal_config_sha256=file_sha256(thermal_path), source_sha256=source_hashes(cfg),
        boundary_csv_sha256=file_sha256(nodes), convection_package_sha256=file_sha256(package_path),
        convection_conditions=package['conditions'], material=dict(name='Tungsten'),
        analysis=dict(duration_s=200, initial_temperature_k=300), settings=dict(h_source='two_endpoint', duration=200),
        final_temperature=dict(maximum_K=503, mean_K=501.5),
        result_sha256={name: file_sha256(output / name) for name in
                       ('wall_temperature_history.csv', 'temperature_history.csv', 'thermal_nodes_final.csv')}))


class EndpointReportTests(unittest.TestCase):
    def test_root_and_effective_configs_validate_nested_recovery_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'nose'; cfg = endpoint_case(work)
            self.assertFalse((work / '04_final.cas.h5').exists())
            self.assertTrue(temperature_complete(cfg))
            self.assertTrue(aero_complete_for_report(cfg))
            effective = copy.deepcopy(cfg)
            effective.update(_endpoint_kind='adiabatic', _study_work_dir=str(work))
            effective['project']['work_dir'] = str(work / 'cfd/adiabatic')
            effective['thermal'] = {'mode': 'adiabatic'}
            effective['temperature_export']['output_directory'] = str(work / 'temperature')
            self.assertEqual(source_work(effective), work / 'cfd/adiabatic')
            self.assertEqual(source_cfg(effective)['project']['work_dir'], str(work / 'cfd/adiabatic'))
            self.assertTrue(temperature_complete(effective))
            self.assertTrue(aero_complete_for_report(effective))
            self.assertEqual(build_gas_row(cfg)['status'], 'complete')
            (work / 'cfd/adiabatic/04_final.dat.h5').write_bytes(b'unsampled new solution')
            self.assertFalse(temperature_complete(cfg))
            self.assertFalse(aero_complete_for_report(cfg))

    def test_derived_h_maps_and_comparison_use_current_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); work = root / 'nose'; endpoint_case(work)
            rows, outputs = create_gas_reports(root, root / 'comparison')
            self.assertEqual(rows[0]['status'], 'complete')
            self.assertTrue(Path(outputs['heat_transfer_coefficient_comparison']).is_file())
            for name in ('derived_h_profile.png', 'derived_h_map.png', 'adiabatic_wall_temperature_map.png',
                         'cold_wall_heat_flux_profile.png', 'cold_wall_heat_flux_map.png'):
                self.assertTrue((work / 'reports' / name).is_file())
            self.assertFalse((work / 'cfd/adiabatic/reports').exists())
            (work / 'cfd/cold_wall/04_final.dat.h5').write_bytes(b'changed calibration')
            _, outputs = create_gas_reports(root, root / 'comparison')
            self.assertNotIn('heat_transfer_coefficient_comparison', outputs)
            self.assertFalse((work / 'reports/derived_h_profile.png').exists())
            self.assertTrue((work / 'reports/adiabatic_wall_temperature_map.png').is_file())

    def test_mechanical_freshness_checks_both_endpoints_and_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); work = root / 'nose'; cfg = endpoint_case(work)
            thermal_path = root / 'thermal.yaml'; thermal_path.write_text('transient: {duration_s: 200}\n')
            solid_result(work, cfg, thermal_path)
            self.assertTrue(mechanical_complete(work, cfg))
            (work / 'cfd/cold_wall/04_final.dat.h5').write_bytes(b'new cold solution')
            self.assertFalse(mechanical_complete(work, cfg))

    def test_spatial_h_differences_do_not_split_comparable_noses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            thermal_path = root / 'thermal.yaml'; thermal_path.write_text('transient: {duration_s: 200}\n')
            for name, scale in (('blunt', 1), ('sharp', 2)):
                work = root / name; cfg = endpoint_case(work, scale)
                solid_result(work, cfg, thermal_path)
            rows, outputs = create_mechanical_reports(root, root / 'comparison/solid')
            self.assertEqual(len(rows), 2)
            self.assertIn('report_manifest', outputs)
            self.assertTrue((root / 'comparison/solid/wall_temperature_history_comparison.png').is_file())
            thermal_path.write_text('transient: {duration_s: 300}\n')
            rows, outputs = create_mechanical_reports(root, root / 'comparison/solid')
            self.assertEqual(rows, [])
            self.assertFalse((root / 'comparison/solid').exists())


if __name__ == '__main__':unittest.main()
