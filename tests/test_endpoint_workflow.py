"""Exercise endpoint sequencing, real field adapters, resume and continuation."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from pipeline.aero import aerodynamic_summary
from pipeline.config import load_config
from pipeline.endpoint_export import export_endpoint
from pipeline.endpoint_loading import build_convection_package
from pipeline.endpoint_workflow import effective_cfg, execute_stage, stage_files
from pipeline.study import BatchError, run_case
from pipeline.temperature_export import export_temperatures, temperature_paths, temperature_conditions
from pipeline.utils import atomic_json, file_sha256
from test_adiabatic_study import field_solver
from test_nose_trade import fluent

ROOT = Path(__file__).resolve().parents[1]
GOOD_LOG = 'iter continuity x-velocity y-velocity energy time/iter\n1161 9.9e-4 1.2e-4 6.0e-5 1.1e-4 0:00:00 0\n'


def endpoint_fields(cold=False):
    solver, calls = field_solver(flux=-17000)
    original = solver.fields.field_data.get_field_data
    def get(request):
        result = original(request)
        if cold and getattr(request,'field_name',None) == 'temperature':
            return {zone:np.full_like(value,300.) for zone,value in result.items()}
        return result
    solver.fields.field_data.get_field_data = get
    return solver, calls


class EndpointWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.work = self.root/'runs/nose'
        self.work.mkdir(parents=True)
        self.cfg = yaml.safe_load((ROOT/'config.yaml').read_text())
        geometry = self.root/'nose.x_t'; geometry.write_text('CAD fixture')
        self.cfg['geometry']['input_file'] = str(geometry)
        self.cfg['project'].update(name='nose',work_dir=str(self.work))
        self.cfg['runtime']['meshing_release_delay_s'] = 0
        self.path = self.work/'_batch/config.yaml'; self.path.parent.mkdir()
        self.save_config()
        self.calls = []
        self.thermal = self.root/'thermal.yaml'
        self.thermal.write_text((ROOT/'thermal_analysis.yaml').read_text())

    def save_config(self):self.path.write_text(yaml.safe_dump(self.cfg))

    def runner(self, command, log_path, **kwargs):
        cfg = load_config(command[command.index('--config')+1])
        stage = command[command.index('--stage')+1]
        self.calls.append(stage)
        work = Path(cfg['project']['work_dir'])
        log_path.parent.mkdir(parents=True,exist_ok=True)
        log_path.write_text(GOOD_LOG)
        if stage == 'mesh':(work/'02_mesh.msh.h5').write_bytes(b'shared mesh')
        elif stage in ('cold_wall','adiabatic'):
            endpoint = effective_cfg(cfg,stage); directory = Path(endpoint['project']['work_dir'])
            directory.mkdir(parents=True,exist_ok=True)
            for name in ('04_final.cas.h5','04_final.dat.h5'):
                (directory/name).write_bytes((stage+name+str(cfg['thermal'])).encode())
            metadata = dict(thermal_mode=endpoint['thermal']['mode'],source_conditions=temperature_conditions(endpoint),
                source_mesh_sha256=file_sha256(work/'02_mesh.msh.h5'),source_geometry_sha256=file_sha256(cfg['geometry']['input_file']),
                body_wall_zones=['body_wall'])
            if stage == 'adiabatic':
                self.assertEqual(json.loads((work/'cfd/cold_wall/summary.json').read_text())['numerical_quality']['status'],'residual_screen_passed')
                metadata['reference_source_sha256']={key:file_sha256(work/'cfd/cold_wall'/name) for key,name in
                    (('final_case','04_final.cas.h5'),('final_data','04_final.dat.h5'))}
            atomic_json(directory/'summary.json',metadata)
        elif stage in ('cold_export','adiabatic_export'):
            cold = stage == 'cold_export'; kind = 'cold_wall' if cold else 'adiabatic'
            endpoint = effective_cfg(cfg,kind); solver,_ = endpoint_fields(cold)
            export_endpoint(solver,endpoint,['body_wall'])
            if not cold:
                result = export_temperatures(solver,endpoint,['body_wall'])
                directory = work/'cfd/adiabatic'
                result.update(source_sha256={key:file_sha256(directory/name) for key,name in
                    (('final_case','04_final.cas.h5'),('final_data','04_final.dat.h5'))},
                    wall_csv_sha256=file_sha256(temperature_paths(endpoint)[0]))
                atomic_json(temperature_paths(endpoint)[1],result)
        elif stage == 'thermal_bc':build_convection_package(cfg)
        elif stage == 'aero':
            endpoint = effective_cfg(cfg,'adiabatic')
            result = aerodynamic_summary(fluent(),endpoint,['body_wall.1','body_wall.2'])
            result['source_sha256']={key:file_sha256(work/'cfd/adiabatic'/name) for key,name in
                (('final_case','04_final.cas.h5'),('final_data','04_final.dat.h5'))}
            atomic_json(work/'aero_summary.json',result)
        else:self.fail(f'Unexpected stage {stage}')

    def study(self, **kwargs):return run_case(self.path,self.thermal,_runner=self.runner,**kwargs)

    def test_full_cfd_calibration_and_aero_reuse_one_mesh_and_resume_without_work(self):
        result = self.study(stop_after='aero')
        self.assertEqual(self.calls,['mesh','cold_wall','cold_export','adiabatic','adiabatic_export','thermal_bc','aero'])
        self.assertEqual(result['status'],'complete')
        self.assertTrue((self.work/'thermal_bc/convection_nodes.csv').is_file())
        self.assertFalse((self.work/'04_final.cas.h5').exists())
        self.calls.clear(); result = self.study(stop_after='aero')
        self.assertFalse(self.calls)
        self.assertEqual(set(result['stages'].values()),{'reused'})

    def test_calibration_setting_changes_reuse_both_flow_fields(self):
        self.study(stop_after='aero'); self.calls.clear()
        self.cfg['workflow']['min_delta_temperature_k'] = 30; self.save_config()
        self.study(stop_after='aero')
        self.assertEqual(self.calls,['thermal_bc'])

    def test_cold_condition_change_retains_mesh_and_archives_both_flow_fields(self):
        self.study(stop_after='aero'); self.calls.clear()
        # Use a boundary-condition preserving value in the virtual exporter.
        self.cfg['solver']['max_iterations'] += 100; self.save_config()
        self.study(stop_after='cold_wall')
        self.assertEqual(self.calls,['cold_wall'])
        self.assertTrue((self.work/'02_mesh.msh.h5').is_file())
        self.assertFalse((self.work/'cfd/adiabatic/04_final.cas.h5').exists())
        self.assertTrue(list((self.work/'_history').glob('*/cfd/adiabatic/04_final.cas.h5')))

    def test_temperature_limiting_blocks_reference_and_keeps_diagnostic_pair(self):
        def bad(command,log_path,**kwargs):
            self.runner(command,log_path,**kwargs)
            if command[command.index('--stage')+1] == 'cold_wall':
                log_path.write_text(GOOD_LOG+'temperature limited to 5.000000e+03 in 10 cells\n')
        with self.assertRaisesRegex(BatchError,'residual screen'):
            run_case(self.path,self.thermal,stop_after='aero',_runner=bad)
        self.assertEqual(self.calls,['mesh','cold_wall'])
        self.assertTrue((self.work/'cfd/cold_wall/04_final.dat.h5').is_file())
        self.assertFalse((self.work/'thermal_bc/convection_nodes.csv').exists())

    def test_adiabatic_stage_uses_shared_solver_and_cold_checkpoint(self):
        with patch('pipeline.solve.solve_case',return_value={'status':'complete'}) as solve:
            execute_stage(load_config(self.path),'adiabatic',self.thermal)
        target, mesh = solve.call_args.args
        cold = solve.call_args.kwargs['reference_cfg']
        self.assertEqual(mesh,self.work/'02_mesh.msh.h5')
        self.assertEqual(target['thermal'],{'mode':'adiabatic'})
        self.assertEqual(cold['thermal'],{'mode':'fixed_temperature','wall_temperature_k':300.})
        self.assertEqual(target['flight'],cold['flight'])
        self.assertEqual(target['solver']['max_iterations'],600)
        self.assertEqual(effective_cfg(target,'cold_wall')['thermal'],cold['thermal'])

    def test_cold_export_uses_face_flux_and_explicit_sign_without_nodal_flux_request(self):
        self.study(stop_after='cold_wall')
        solver,calls = endpoint_fields(True)
        export_endpoint(solver,effective_cfg(load_config(self.path),'cold_wall'),['body_wall'])
        flux = [r for r in calls if getattr(r,'field_name',None) == 'heat-flux']
        self.assertEqual(len(flux),1)
        self.assertFalse(flux[0].node_value)
        self.assertTrue(flux[0].boundary_value)
        from pipeline.reporting import read_csv
        rows = read_csv(self.work/'cfd/cold_wall/thermal_bc/wall_faces.csv')
        self.assertTrue(all(float(r['heat_flux_into_wall_W_m2']) == 17000 for r in rows))

    def test_boundary_walk_is_stable_when_native_face_order_and_orientation_change(self):
        from pipeline.thermal_export import order_segments
        xy=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
        records=[dict(start=xy[i],end=xy[(i+1)%4],sample=10+i) for i in range(4)]
        original=order_segments(records)
        reordered=[dict(start=r['end'],end=r['start'],sample=r['sample']) for r in records[::-1]]
        self.assertEqual([r['sample'] for r in order_segments(reordered)],[r['sample'] for r in original])


if __name__ == '__main__':unittest.main()
