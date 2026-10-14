"""Single CFD solve through prescribed-temperature Mechanical, without engines."""
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
from pipeline.study import BatchError, run_case, stage_files, stage_fingerprint, output_hashes
from pipeline.temperature_export import export_temperatures, temperature_conditions, temperature_paths
from pipeline.thermal import run_thermal_analysis
from pipeline.utils import atomic_json, file_sha256
from test_adiabatic_study import field_solver
from test_nose_trade import fluent

ROOT=Path(__file__).resolve().parents[1]
GOOD_LOG='iter continuity x-velocity y-velocity energy time/iter\n1161 9.9e-4 1.2e-4 6.0e-5 1.1e-4 0:00:00 0\n'


class SingleWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.work=self.root/'runs/nose'; self.work.mkdir(parents=True)
        self.cfg=yaml.safe_load((ROOT/'config.yaml').read_text())
        self.cfg['project'].update(name='nose',work_dir=str(self.work))
        cad=self.root/'nose.x_t'; cad.write_text('CAD fixture'); self.cfg['geometry']['input_file']=str(cad)
        self.cfg['runtime']['meshing_release_delay_s']=0
        self.path=self.work/'_batch/config.yaml'; self.path.parent.mkdir(); self.write_config()
        ta=yaml.safe_load((ROOT/'thermal_analysis.yaml').read_text()); ta['visualization']['enabled']=False
        self.thermal=self.root/'thermal.yaml'; self.thermal.write_text(yaml.safe_dump(ta))
        self.calls=[]

    def write_config(self):self.path.write_text(yaml.safe_dump(self.cfg))

    def runner(self, command, log_path, **kwargs):
        cfg=load_config(command[command.index('--config')+1])
        entry=Path(command[2]).name; self.calls.append(entry)
        self.assertNotIn('--stage',command)
        self.assertNotIn('_endpoint_kind',cfg)
        work=Path(cfg['project']['work_dir'])
        self.assertEqual(work,self.work)
        log_path.parent.mkdir(parents=True,exist_ok=True); log_path.write_text(GOOD_LOG)
        if entry=='run_all.py':(work/'02_mesh.msh.h5').write_bytes(b'unchanged shared mesh')
        elif entry=='run_solver_only.py':
            for name in ('04_final.cas.h5','04_final.dat.h5'):
                (work/name).write_bytes((name+str(cfg['thermal'])).encode())
            atomic_json(work/'summary.json',dict(thermal_mode='radiative_equilibrium',source_conditions=temperature_conditions(cfg),
                source_geometry_sha256=file_sha256(cfg['geometry']['input_file']),source_mesh_sha256=file_sha256(work/'02_mesh.msh.h5'),
                body_wall_zones=['body_wall'],wall_thermal_setup={'body_wall':{'thermal_condition':'Radiation','emissivity':.9,
                    'external_radiation_temperature_k':3}}))
        elif entry=='export_temperature_only.py':
            result=export_temperatures(field_solver()[0],cfg,['body_wall'])
            result.update(source_sha256={key:file_sha256(work/name) for key,name in
                (('final_case','04_final.cas.h5'),('final_data','04_final.dat.h5'))},
                wall_csv_sha256=file_sha256(temperature_paths(cfg)[0]))
            atomic_json(temperature_paths(cfg)[1],result)
        elif entry=='export_aero_only.py':
            result=aerodynamic_summary(fluent(),cfg,['body_wall.1','body_wall.2'])
            result['source_sha256']={key:file_sha256(work/name) for key,name in
                (('final_case','04_final.cas.h5'),('final_data','04_final.dat.h5'))}
            atomic_json(work/'aero_summary.json',result)
        elif entry=='run_thermal_analysis.py':
            ta=command[command.index('--thermal-config')+1]
            def solve(engine,load,settings,size):
                self.assertEqual(load.mode,'prescribed_temperature')
                self.assertFalse(settings['include_rad']); self.assertNotIn('h',settings)
                xyz=np.c_[np.vstack([load.xy,load.xy.mean(axis=0)]),np.zeros(len(load.xy)+1)]
                values=[np.r_[load.surface_temperature,400.],np.r_[load.surface_temperature,800.]]
                load.constrained_node_count=len(load.xy)
                return np.arange(1,len(xyz)+1),xyz,values,[2.,settings['duration']],2
            with patch('ansys.mapdl.core.launch_mapdl',return_value=SimpleNamespace(exit=lambda:None)), \
                 patch('pipeline.thermal.solve_thermal',side_effect=solve):
                run_thermal_analysis(cfd_config=self.path,thermal_config=ta)
        else:self.fail('Unexpected stage '+entry)

    def study(self,**kwargs):return run_case(self.path,self.thermal,_runner=self.runner,**kwargs)

    def test_default_run_requires_one_solver_and_separate_prescribed_temperature_analysis(self):
        result=self.study()
        self.assertEqual(self.calls,['run_all.py','run_solver_only.py','export_temperature_only.py','export_aero_only.py','run_thermal_analysis.py'])
        self.assertEqual(result['status'],'complete')
        self.assertEqual(json.loads((self.work/'_batch/stage_state.json').read_text())['strategy'],'single_solve')
        self.assertFalse((self.work/'cfd').exists())
        self.assertFalse((self.work/'thermal_bc').exists())
        summary=json.loads((self.work/'thermal_analysis/thermal_summary.json').read_text())
        self.assertEqual(summary['model_version'],'5.0-prescribed-temperature')
        self.assertIsNone(summary['net_heat_load_per_unit_span_J_m'])
        self.assertEqual(summary['analysis']['duration_s'],200)
        self.calls.clear(); second=self.study()
        self.assertFalse(self.calls)
        self.assertEqual(set(second['stages'].values()),{'reused'})

    def test_exposure_change_reuses_cfd_and_export_and_only_runs_mechanical(self):
        self.study(); self.calls.clear()
        ta=yaml.safe_load(self.thermal.read_text());ta['transient']['duration_s']=150
        self.thermal.write_text(yaml.safe_dump(ta))
        self.study()
        self.assertEqual(self.calls,['run_thermal_analysis.py'])

    def test_external_radiation_change_reuses_mesh_and_reruns_single_solver(self):
        self.study(stop_after='aero');self.calls.clear()
        self.cfg['thermal']['emissivity']=.8;self.write_config()
        self.study(stop_after='aero')
        self.assertEqual(self.calls,['run_solver_only.py','export_temperature_only.py','export_aero_only.py'])

    def test_old_endpoint_mesh_is_reused_and_endpoint_files_are_preserved_as_history(self):
        mesh=self.work/'02_mesh.msh.h5';mesh.write_bytes(b'existing valid mesh')
        cfg=load_config(self.path);files=stage_files(self.work,cfg)
        old=copy.deepcopy(cfg);old['workflow']={'strategy':'two_endpoint'};old['thermal']={'mode':'fixed_temperature','wall_temperature_k':300}
        state=dict(version=3,strategy='two_endpoint',stages={'mesh':dict(status='complete',
            fingerprint=stage_fingerprint('mesh',old,{},files),output_sha256=output_hashes(files['mesh']))})
        atomic_json(self.work/'_batch/stage_state.json',state)
        endpoint=self.work/'cfd/cold_wall/04_final.dat.h5';endpoint.parent.mkdir(parents=True);endpoint.write_bytes(b'old cold data')
        self.study(stop_after='solver')
        self.assertEqual(self.calls,['run_solver_only.py'])
        self.assertEqual(mesh.read_bytes(),b'existing valid mesh')
        saved=list((self.work/'_history').glob('*/cfd/cold_wall/04_final.dat.h5'))
        self.assertEqual(len(saved),1);self.assertEqual(saved[0].read_bytes(),b'old cold data')

    def test_failed_radiative_residual_screen_retains_pair_and_blocks_mechanical(self):
        def bad(command,log_path,**kwargs):
            self.runner(command,log_path,**kwargs)
            if Path(command[2]).name=='run_solver_only.py':log_path.write_text(GOOD_LOG+'temperature limited to 5.000000e+03 in 10 cells\n')
        with self.assertRaisesRegex(BatchError,'did not converge'):
            run_case(self.path,self.thermal,_runner=bad)
        self.assertEqual(self.calls,['run_all.py','run_solver_only.py'])
        self.assertTrue((self.work/'04_final.dat.h5').is_file())
        self.assertFalse((self.work/'thermal_analysis').exists())

    def test_repaired_startup_reuses_mesh_and_archives_diverged_solution_before_fresh_solve(self):
        from pipeline.migration import migrate_config
        self.cfg['solver'].update(initialization='hybrid',courant_number=None,
                                  flow_scheme=None,max_iterations=1200)
        self.write_config()
        def diverged(command,log_path,**kwargs):
            self.runner(command,log_path,**kwargs)
            if Path(command[2]).name=='run_solver_only.py':
                log_path.write_text('iter continuity x-velocity y-velocity energy time/iter\n'
                    '1200 2.8567e10 6.1723e14 6.1523e15 3.7669e20 0:00:00 0\n'
                    'temperature limited to 5.000000e+03 in 1264 cells\n')
        with self.assertRaisesRegex(BatchError,'did not converge'):
            run_case(self.path,self.thermal,_runner=diverged)
        mesh_bytes=(self.work/'02_mesh.msh.h5').read_bytes()
        failed_data=(self.work/'04_final.dat.h5').read_bytes()
        self.assertFalse((self.work/'temperature').exists())
        self.assertTrue(migrate_config(self.path))
        repaired=load_config(self.path)
        self.assertEqual(repaired['solver']['courant_number'],.3)
        self.assertEqual(repaired['solver']['initialization'],'freestream')
        self.assertEqual(repaired['thermal'],self.cfg['thermal'])
        self.calls.clear()
        result=self.study(stop_after='solver')
        self.assertEqual(self.calls,['run_solver_only.py'])
        self.assertEqual(result['stages']['mesh'],'reused')
        self.assertEqual((self.work/'02_mesh.msh.h5').read_bytes(),mesh_bytes)
        retained=list((self.work/'_history').glob('*/04_final.dat.h5'))
        self.assertEqual(len(retained),1)
        self.assertEqual(retained[0].read_bytes(),failed_data)
        state=json.loads((self.work/'_batch/stage_state.json').read_text())
        self.assertEqual(state['stages']['solver']['status'],'complete')
        self.assertNotEqual(state['stages'].get('thermal',{}).get('status'),'complete')
        self.assertFalse((self.work/'thermal_analysis').exists())


if __name__=='__main__':unittest.main()
