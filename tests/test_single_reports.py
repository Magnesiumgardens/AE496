"""Radiating CFD reports and prescribed-surface Mechanical provenance."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from pipeline.aero import aerodynamic_summary
from pipeline.gas_reporting import aero_complete_for_report, create_gas_reports, temperature_complete
from pipeline.mechanical_reporting import create_mechanical_reports, mechanical_complete
from pipeline.reporting import create_thermal_visualizations, read_csv, write_csv
from pipeline.temperature_export import export_temperatures, temperature_paths
from pipeline.utils import atomic_json, file_sha256
from test_adiabatic_study import config, field_solver
from test_nose_trade import fluent


def saved_single(work):
    cfg = config(work)
    cfg['workflow'] = {'strategy': 'single_solve'}
    cfg['thermal'] = {'mode': 'radiative_equilibrium', 'wall_temperature_k': 300.,
                      'emissivity': .9, 'external_radiation_temperature_k': 3.}
    (work / '_batch').mkdir(parents=True)
    Path(cfg['geometry']['input_file']).write_bytes(b'CAD fixture')
    (work / '02_mesh.msh.h5').write_bytes(b'planar mesh fixture')
    (work / '_batch/config.yaml').write_text(yaml.safe_dump(cfg))
    for name in ('04_final.cas.h5', '04_final.dat.h5'):
        (work / name).write_bytes((work.name + name).encode())
    source = {key: file_sha256(work / name) for key, name in
              (('final_case', '04_final.cas.h5'), ('final_data', '04_final.dat.h5'))}
    atomic_json(work / 'summary.json', dict(thermal_mode='radiative_equilibrium',
        source_geometry_sha256=file_sha256(cfg['geometry']['input_file']),
        source_mesh_sha256=file_sha256(work / '02_mesh.msh.h5'),
        numerical_quality=dict(status='residual_screen_passed', source_sha256=source)))
    metadata = export_temperatures(field_solver()[0], cfg, ['body_wall'])
    metadata.update(source_sha256=source, wall_csv_sha256=file_sha256(temperature_paths(cfg)[0]))
    atomic_json(temperature_paths(cfg)[1], metadata)
    aero = aerodynamic_summary(fluent(), cfg, ['body_wall.1', 'body_wall.2'])
    aero['source_sha256'] = source
    atomic_json(work / 'aero_summary.json', aero)
    return cfg


def saved_prescribed(work, cfg, thermal_path):
    output = work / 'thermal_analysis'; output.mkdir()
    xy = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    wall = [dict(time_s=time, surface_order_index=i, x_m=x, y_m=y,
                 wall_temperature_K=300 if time == 0 else 2000 + 100 * i,
                 loading_state='initial_before_application' if time == 0 else 'prescribed_surface')
            for time in (0, 100, 200) for i, (x, y) in enumerate(xy)]
    write_csv(output / 'wall_temperature_history.csv', wall)
    write_csv(output / 'temperature_history.csv', [dict(time_s=time, minimum_K=300+time,
        mean_K=300+2*time, maximum_K=2300 if time else 300) for time in (0, 100, 200)])
    write_csv(output / 'thermal_nodes_final.csv', [dict(node_id=i+1,x_m=x,y_m=y,temperature_K=2000+100*i)
                                                 for i,(x,y) in enumerate(xy)])
    nodes = temperature_paths(cfg)[0].parent / 'wall_temperature_nodes.csv'
    atomic_json(output / 'thermal_summary.json', dict(status='complete',
        model_version='5.0-prescribed-temperature', loading_mode='prescribed_temperature',
        thermal_config_path=str(thermal_path), thermal_config_sha256=file_sha256(thermal_path),
        source_sha256={key: file_sha256(work / name) for key, name in
                      (('final_case','04_final.cas.h5'), ('final_data','04_final.dat.h5'))},
        boundary_csv_sha256=file_sha256(nodes),
        source_temperature_summary_sha256=file_sha256(temperature_paths(cfg)[1]),
        material=dict(name='Tungsten'), analysis=dict(duration_s=200,initial_temperature_k=300),
        settings=dict(loading_mode='prescribed_temperature',duration=200),
        final_temperature=dict(maximum_K=2300,mean_K=700),
        result_sha256={name:file_sha256(output/name) for name in
                       ('wall_temperature_history.csv','temperature_history.csv','thermal_nodes_final.csv')}))


class SingleSolveReportTests(unittest.TestCase):
    def test_radiating_reports_need_no_endpoints_and_remove_previous_calibration_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('blunt','sharp'):
                work=root/name;cfg=saved_single(work)
                report=work/'reports';report.mkdir()
                for old in ('derived_h_profile.png','cold_wall_heat_flux_map.png','adiabatic_wall_temperature_map.png'):
                    (report/old).write_bytes(b'old model')
                self.assertTrue(temperature_complete(cfg))
                self.assertFalse((work/'cfd').exists())
            rows,files=create_gas_reports(root,root/'comparison')
            self.assertEqual([row['thermal_mode'] for row in rows],['radiative_equilibrium']*2)
            self.assertTrue(all(row['status']=='complete' for row in rows))
            self.assertNotIn('heat_transfer_coefficient_comparison',files)
            self.assertEqual(Path(files['drag_vs_temperature']).name,'drag_vs_wall_temperature.png')
            for name in ('blunt','sharp'):
                report=root/name/'reports'
                self.assertTrue((report/'radiative_equilibrium_wall_temperature_map.png').is_file())
                self.assertFalse((report/'derived_h_profile.png').exists())
                self.assertFalse((report/'cold_wall_heat_flux_map.png').exists())
                self.assertFalse((report/'adiabatic_wall_temperature_map.png').exists())
                manifest=json.loads((report/'radiative_equilibrium_case_report.json').read_text())
                self.assertIn('equilibrium',manifest['wall_boundary_condition'])
                self.assertIn('Separate Mechanical loading',manifest['interpretation'])
            self.assertFalse((root/'comparison/adiabatic_report.json').exists())
            manifest=json.loads((root/'comparison/radiative_equilibrium_report.json').read_text())
            self.assertTrue(manifest['comparable_conditions'])
            for stem, field in (("raw_max_wall_temperature_comparison", "maximum_wall_temperature_K"),
                                ("raw_max_near_wall_fluid_temperature_comparison", "maximum_near_wall_fluid_temperature_K")):
                self.assertTrue(Path(files[stem + "_png"]).is_file())
                self.assertEqual(manifest["files"][stem + "_png"], files[stem + "_png"])
                records = read_csv(files[stem + "_csv"])
                self.assertEqual([record["design"] for record in records], ["blunt", "sharp"])
                for record, summary in zip(records, rows):
                    self.assertEqual(float(record[field]), summary[field])

    def test_old_adiabatic_export_and_unscreened_case_cannot_supply_radiating_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)/'nose';cfg=saved_single(work)
            path=temperature_paths(cfg)[1];metadata=json.loads(path.read_text())
            metadata['thermal_mode']='adiabatic';atomic_json(path,metadata)
            self.assertFalse(temperature_complete(cfg))
            metadata['thermal_mode']='radiative_equilibrium';atomic_json(path,metadata)
            atomic_json(work/'summary.json',{'thermal_mode':'radiative_equilibrium'})
            self.assertFalse(temperature_complete(cfg))

    def test_geometry_mesh_and_node_changes_cannot_reuse_radiating_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)/'nose';cfg=saved_single(work)
            self.assertTrue(aero_complete_for_report(cfg))
            nodes=temperature_paths(cfg)[0].parent/'wall_temperature_nodes.csv'
            original=nodes.read_bytes();nodes.write_bytes(original+b'\n')
            self.assertFalse(temperature_complete(cfg))
            nodes.write_bytes(original)
            cad=Path(cfg['geometry']['input_file']);cad.write_bytes(b'changed geometry')
            self.assertFalse(temperature_complete(cfg))
            self.assertFalse(aero_complete_for_report(cfg))
            cad.write_bytes(b'CAD fixture')
            (work/'02_mesh.msh.h5').write_bytes(b'changed mesh')
            self.assertFalse(temperature_complete(cfg))
            self.assertFalse(aero_complete_for_report(cfg))

    def test_prescribed_mechanical_reports_check_load_metadata_and_label_surface_as_imposed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);work=root/'nose';cfg=saved_single(work)
            thermal=root/'thermal.yaml';thermal.write_text('loading: {mode: prescribed_temperature}\n')
            saved_prescribed(work,cfg,thermal)
            self.assertTrue(mechanical_complete(work,cfg))
            rows,files=create_mechanical_reports(root,root/'comparison/solid')
            self.assertEqual(len(rows),1)
            manifest=json.loads(Path(files['report_manifest']).read_text())
            self.assertEqual(manifest['prescribed_surface_cases'],['nose'])
            self.assertTrue(Path(files['solid_temperature_history_comparison']).is_file())
            self.assertFalse((work/'reports/final_wall_heat_flux_map.png').exists())
            wall=read_csv(work/'thermal_analysis/wall_temperature_history.csv')
            self.assertEqual([r['wall_temperature_K'] for r in wall if r['time_s']=='100'],
                             [r['wall_temperature_K'] for r in wall if r['time_s']=='200'])
            metadata=temperature_paths(cfg)[1];metadata.write_text(metadata.read_text()+'\n')
            self.assertFalse(mechanical_complete(work,cfg))

    def test_prescribed_visualization_omits_and_clears_heat_flux_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);viz=output/'visualization';viz.mkdir()
            for name in ('applied_heat_flux_map.png','applied_heat_flux_position.png'):(viz/name).write_bytes(b'stale')
            xy=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
            xyz=np.c_[np.vstack([xy,[.5,.5]]),np.zeros(5)]
            snapshots=[dict(time_s=200.,temperature_K=np.array([2000.,2100.,2200.,2300.,1800.]))]
            history=[dict(time_s=200.,minimum_K=1800.,mean_K=2080.,maximum_K=2300.)]
            files=create_thermal_visualizations(output,xy,np.zeros(4),xyz,history,snapshots,
                dict(loading_mode='prescribed_temperature',save_heat_flux_image=True,make_gif=False))
            self.assertTrue(Path(files['final_temperature_field_png']).is_file())
            self.assertFalse((viz/'applied_heat_flux_map.png').exists())
            self.assertFalse((viz/'applied_heat_flux_position.png').exists())
            self.assertFalse(any('heat_flux' in key for key in files))


if __name__=='__main__':unittest.main()
