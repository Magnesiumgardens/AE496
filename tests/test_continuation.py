"""Saved-flow continuation, provenance, preservation, and convergence gates."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from pipeline.config import load_config
from pipeline.continuation import continue_case, continue_solver, preflight, source_hashes
from pipeline.solution_quality import saved_quality
from pipeline.study import BatchError, reusable, stage_files, stage_fingerprint
from pipeline.temperature_export import atomic_csv
from pipeline.utils import atomic_json, file_sha256
from test_solver_numerics import ServerDictionary

ROOT=Path(__file__).resolve().parents[1]


def sample(iteration):
    return dict(iteration=iteration,wall_temperature_min_k=1000.,wall_temperature_avg_k=2000.,wall_temperature_max_k=2500.)


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.work=self.root/'runs/nose';self.work.mkdir(parents=True)
        self.path=self.work/'_batch/config.yaml';self.path.parent.mkdir()
        geometry=self.root/'nose.x_t';geometry.write_text('CAD fixture')
        cfg=yaml.safe_load((ROOT/'config.yaml').read_text())
        cfg['project']=dict(name='nose',work_dir=str(self.work));cfg['geometry']['input_file']=str(geometry)
        self.path.write_text(yaml.safe_dump(cfg));self.cfg=load_config(self.path)
        for name in ('02_mesh.msh.h5','04_final.cas.h5','04_final.dat.h5'):(self.work/name).write_bytes(('old '+name).encode())
        self.summary=dict(thermal_mode='adiabatic',body_wall_zones=['body_wall'],
            numerical_quality={'status':'failed','recent_temperature_or_pressure_limiting':False,'source_sha256':source_hashes(self.work)})
        atomic_json(self.work/'summary.json',self.summary)
        atomic_csv(self.work/'solver_temperature_history.csv',[sample(100),sample(1200)])
        self.summary['temperature_history']={'status':'complete','csv_sha256':file_sha256(self.work/'solver_temperature_history.csv'),
                                             'source_sha256':source_hashes(self.work)}
        atomic_json(self.work/'summary.json',self.summary)
        files=stage_files(self.work,self.cfg)
        atomic_json(self.work/'_batch/stage_state.json',{'stages':{'solver':{'status':'failed','fingerprint':stage_fingerprint('solver',self.cfg,{},files)}}})
        self.calls=[]
    def tearDown(self):self.tmp.cleanup()

    def runner(self,command,log_path,**kwargs):
        self.calls.append(command)
        cfg=load_config(self.path)
        staged=Path(command[command.index('--output-dir')+1]);staged.mkdir(exist_ok=True)
        for name in ('04_final.cas.h5','04_final.dat.h5'):(staged/name).write_bytes(('continued '+name).encode())
        atomic_csv(staged/'solver_temperature_history.csv',[sample(1225),sample(1250)])
        summary=copy.deepcopy(self.summary)
        summary['temperature_history']={'status':'complete','csv':'solver_temperature_history.csv','sample_count':2,
            'source_sha256':source_hashes(staged),'csv_sha256':file_sha256(staged/'solver_temperature_history.csv')}
        atomic_json(staged/'summary.json',summary)
        log_path.write_text('iter continuity x-velocity y-velocity energy time/iter\n1250 5e-4 5e-4 5e-4 5e-7 0:00:00 0\n')

    def test_continuation_keeps_previous_pair_and_marks_passed_stage_reusable(self):
        result=continue_case(self.path,50,_runner=self.runner)
        self.assertEqual(result['status'],'complete')
        backup=Path(result['archive'])
        self.assertEqual((backup/'04_final.dat.h5').read_bytes(),b'old 04_final.dat.h5')
        self.assertEqual((self.work/'02_mesh.msh.h5').read_bytes(),b'old 02_mesh.msh.h5')
        history=(self.work/'solver_temperature_history.csv').read_text()
        self.assertIn('1200',history);self.assertIn('1250',history)
        state=json.loads((self.work/'_batch/stage_state.json').read_text())
        files=stage_files(self.work,self.cfg)
        self.assertTrue(reusable(state['stages']['solver'],stage_fingerprint('solver',self.cfg,{},files),files['solver']))
        self.assertEqual(saved_quality(self.cfg)['status'],'residual_screen_passed')
        self.assertEqual(len(self.calls),1)

    def test_still_unconverged_flow_retains_new_progress_and_blocks_downstream(self):
        def runner(command,log_path,**kwargs):
            self.runner(command,log_path,**kwargs)
            log_path.write_text('iter continuity x-velocity y-velocity energy time/iter\n1250 .1 .1 .1 .1 0:00:00 0\n')
        result=continue_case(self.path,50,_runner=runner)
        self.assertEqual(result['status'],'not_converged')
        self.assertEqual(saved_quality(self.cfg)['status'],'failed')
        self.assertEqual((self.work/'04_final.dat.h5').read_bytes(),b'continued 04_final.dat.h5')
        self.assertFalse((self.work/'temperature/wall_temperatures.csv').exists())

    def test_native_failure_preserves_old_pair_and_rolls_back_courant_change(self):
        before=self.path.read_bytes()
        def runner(*args,**kwargs):raise RuntimeError('Native solver failed')
        with self.assertRaisesRegex(RuntimeError,'Native solver failed'):
            continue_case(self.path,50,courant=.3,_runner=runner)
        self.assertEqual((self.work/'04_final.dat.h5').read_bytes(),b'old 04_final.dat.h5')
        self.assertEqual(self.path.read_bytes(),before)
        preflight(load_config(self.path))

    def test_missing_data_output_never_replaces_original_case(self):
        def runner(command,log_path,**kwargs):
            self.runner(command,log_path,**kwargs)
            Path(command[command.index('--output-dir')+1],'04_final.dat.h5').unlink()
        with self.assertRaisesRegex(BatchError,'no usable'):
            continue_case(self.path,50,_runner=runner)
        self.assertEqual((self.work/'04_final.cas.h5').read_bytes(),b'old 04_final.cas.h5')

    def test_changed_saved_data_or_requested_geometry_is_rejected_before_launch(self):
        (self.work/'04_final.dat.h5').write_bytes(b'changed data')
        with self.assertRaisesRegex(BatchError,'provenance'):preflight(self.cfg)
        (self.work/'04_final.dat.h5').write_bytes(b'old 04_final.dat.h5')
        Path(self.cfg['geometry']['input_file']).write_text('changed CAD')
        with self.assertRaisesRegex(BatchError,'do not match'):preflight(self.cfg)

    def test_temperature_limiting_is_rejected_before_continuation(self):
        self.summary['numerical_quality']['recent_temperature_or_pressure_limiting']=True
        atomic_json(self.work/'summary.json',self.summary)
        with self.assertRaisesRegex(BatchError,'limiting'):preflight(self.cfg)

    def test_courant_change_is_persisted_only_with_successful_continuation(self):
        result=continue_case(self.path,50,courant=.3,_runner=self.runner)
        self.assertEqual(load_config(self.path)['solver']['courant_number'],.3)
        self.assertEqual(yaml.safe_load((Path(result['archive'])/'source_config.yaml').read_text())['solver']['courant_number'],.1)

    def test_overlapping_native_iteration_labels_remain_separate(self):
        def runner(command,log_path,**kwargs):
            self.runner(command,log_path,**kwargs)
            staged=Path(command[command.index('--output-dir')+1])
            atomic_csv(staged/'solver_temperature_history.csv',[sample(25),sample(50)])
        continue_case(self.path,50,_runner=runner)
        summary=json.loads((self.work/'summary.json').read_text())
        self.assertIn('overlap',summary['temperature_history']['history_scope'])
        self.assertNotIn('1200',(self.work/'solver_temperature_history.csv').read_text())

    def test_native_continuation_never_initializes_or_reads_mesh(self):
        staged=self.work/'_continuation/native-test';staged.mkdir(parents=True)
        calls=[]
        def read_case(file_name):calls.append(('read',file_name))
        def write_case(file_name):
            calls.append(('write',file_name));Path(file_name).write_bytes(b'new case');Path(file_name.replace('.cas.h5','.dat.h5')).write_bytes(b'new data')
        schemes=ServerDictionary({'amg-c':'first-order-upwind'},{'amg-c':['first-order-upwind']})
        solver=SimpleNamespace(exit=lambda:calls.append(('exit',)),settings=SimpleNamespace(
            file=SimpleNamespace(read_case_data=read_case,write_case_data=write_case),
            setup=SimpleNamespace(general=SimpleNamespace(solver=SimpleNamespace(time=SimpleNamespace(get_state=lambda:'steady')))),
            mesh=SimpleNamespace(quality=lambda:calls.append(('quality',))),
            solution=SimpleNamespace(controls=SimpleNamespace(),methods=SimpleNamespace(spatial_discretization=SimpleNamespace(discretization_scheme=schemes)))))
        def iterate(solver,cfg,walls):
            calls.append(('iterate',cfg['solver']['max_iterations']));return {'status':'disabled'}
        with patch('pipeline.continuation.launch_solver',return_value=solver),patch('pipeline.continuation.validate_saved_conditions'),patch('pipeline.continuation.iterate_with_history',side_effect=iterate),patch('pipeline.continuation.configure_residual_criteria',return_value={'energy':1e-6}):
            summary=continue_solver(self.cfg,50,staged)
        self.assertEqual([call[0] for call in calls],['read','quality','iterate','write','exit'])
        self.assertFalse(summary['continuation']['initialization_performed'])
        self.assertEqual(calls[2][1],50)
        self.assertEqual((self.work/'04_final.dat.h5').read_bytes(),b'old 04_final.dat.h5')

    def test_unstable_trial_keeps_previous_flow_and_config_active(self):
        before=self.path.read_bytes()
        def runner(command,log_path,**kwargs):
            self.runner(command,log_path,**kwargs)
            with log_path.open('a') as handle:handle.write('temperature limited to 5000 in 100 cells\n')
        result=continue_case(self.path,50,courant=.3,_runner=runner)
        self.assertEqual(result['status'],'unstable')
        self.assertEqual((self.work/'04_final.dat.h5').read_bytes(),b'old 04_final.dat.h5')
        self.assertEqual(self.path.read_bytes(),before)
        self.assertTrue((Path(result['diagnostics'])/'residual_history.csv').is_file())
        preflight(load_config(self.path))

    def test_locked_optional_raw_history_does_not_rollback_solution(self):
        from pipeline.continuation import copy_atomic
        def runner(command,log_path,**kwargs):
            self.runner(command,log_path,**kwargs)
            (Path(command[command.index('--output-dir')+1])/'solver_temperature_history.out').write_text('raw history')
        def copy(source,target):
            if str(source).endswith('.out'):raise PermissionError('WinError 32: raw history is in use')
            return copy_atomic(source,target)
        with patch('pipeline.continuation.copy_atomic',side_effect=copy):
            result=continue_case(self.path,50,_runner=runner)
        self.assertEqual(result['status'],'complete')
        self.assertEqual((self.work/'04_final.dat.h5').read_bytes(),b'continued 04_final.dat.h5')
        summary=json.loads((self.work/'summary.json').read_text())
        self.assertIn('raw_copy_warning',summary['temperature_history'])
        self.assertTrue(Path(summary['temperature_history']['raw_report_file']).is_file())
        self.assertTrue((Path(summary['temperature_history']['raw_report_file']).parent/'04_final.dat.h5').is_file())

    def test_critical_promotion_error_keeps_candidate_pair_recoverable(self):
        from pipeline.continuation import copy_atomic
        def copy(source,target):
            if str(target).endswith('04_final.dat.h5'):raise PermissionError('target data is locked')
            return copy_atomic(source,target)
        with patch('pipeline.continuation.copy_atomic',side_effect=copy):
            with self.assertRaises(PermissionError):continue_case(self.path,50,_runner=self.runner)
        self.assertEqual((self.work/'04_final.cas.h5').read_bytes(),b'old 04_final.cas.h5')
        staged=next((self.work/'_continuation').iterdir())
        self.assertEqual((staged/'04_final.cas.h5').read_bytes(),b'continued 04_final.cas.h5')
        self.assertEqual((staged/'04_final.dat.h5').read_bytes(),b'continued 04_final.dat.h5')
