"""Coupling and numerical validity tests without Ansys server processes."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from pipeline.convection_loading import ConvectionLoad, read_load, validate_settings
from pipeline.thermal import apply_surface_loads, save_results
from pipeline.thermal_loading import ThermalAnalysisError
from pipeline.solution_quality import assess_log
from pipeline.utils import atomic_json
from pipeline.iteration_history import parse_report_file

ROOT = Path(__file__).resolve().parents[1]


class ConvectionTests(unittest.TestCase):
    def test_h_is_required_and_duration_is_configurable(self):
        cfg = yaml.safe_load((ROOT/'thermal_analysis.yaml').read_text())
        cfd = {'thermal':{'mode':'adiabatic'}}
        for value in (None,0,-1,float('nan')):
            cfg['loading']['heat_transfer_coefficient_w_m2k']=value
            with self.assertRaises(ThermalAnalysisError):validate_settings(cfd,cfg)
        cfg['loading']['heat_transfer_coefficient_w_m2k']=42
        cfg['transient']['duration_s']=180
        result = validate_settings(cfd,cfg)
        self.assertEqual((result['h'],result['duration']),(42,180))
        self.assertFalse(result['include_rad'])

    def test_mechanical_uses_convection_temperature_and_never_cfd_flux(self):
        load = ConvectionLoad(np.array([[0,0],[1,0],[0,1]]),np.array([1000,1200,1400]),42)
        calls=[]
        apply_surface_loads(SimpleNamespace(sfl=lambda *args:calls.append(args)),[11,12,13],load,{'include_rad':False})
        self.assertEqual(calls[0],(11,'CONV',42,42,1000,1200))
        self.assertEqual(calls[-1],(13,'CONV',42,42,1400,1000))
        self.assertTrue(all(call[1]=='CONV' for call in calls))
        np.testing.assert_allclose(load.incoming_flux([300,300,1500]),[29400,37800,-4200])

    def test_reversed_boundary_keeps_temperature_attached_to_coordinate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'nodes.csv'
            path.write_text('surface_order_index,x_m,y_m,recovery_temperature_K\n0,0,0,1000\n1,0,1,1200\n2,1,0,1400\n')
            load=read_load(path,42)
            self.assertEqual(dict(zip(map(tuple,load.xy),load.gas_temperature)),{(0,0):1000,(0,1):1200,(1,0):1400})
            path.write_text('surface_order_index,x_m,y_m,recovery_temperature_K\n0,0,0,1000\n0,0,1,1200\n2,1,0,1400\n')
            with self.assertRaises(ThermalAnalysisError):read_load(path,42)

    def test_saved_solid_history_uses_seconds_and_dynamic_convection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            xy=np.array([[0.,0.],[1.,0.],[0.,1.]])
            load=ConvectionLoad(xy,np.array([1000.,1200.,1400.]),42)
            cfg=yaml.safe_load((ROOT/'thermal_analysis.yaml').read_text())
            cfg['loading']['heat_transfer_coefficient_w_m2k']=42
            cfg['transient']['duration_s']=2
            cfg['transient'].update(initial_time_step_s=1,min_time_step_s=.1,max_time_step_s=2)
            settings=validate_settings({'thermal':{'mode':'adiabatic'}},cfg)
            history,_,_,_=save_results(root,load,[1,2,3],np.c_[xy,np.zeros(3)],[[400,500,600],[600,700,800]],[1,2],settings,.01)
            self.assertEqual([r['time_s'] for r in history],[0,1,2])
            with np.load(root/'thermal_snapshots.npz') as arrays:
                self.assertNotIn('boundary_flux_W_m2',arrays)
                np.testing.assert_equal(arrays['boundary_gas_temperature_K'],load.gas_temperature)


class ResidualTests(unittest.TestCase):
    def assess(self,text):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'solver.log';path.write_text(text)
            result=assess_log(path,{})
            atomic_json(Path(directory)/'result.json',result)
            return result

    def test_real_failed_tail_is_rejected(self):
        result=self.assess(' iter continuity x-velocity y-velocity energy time/iter\n1199 2.7927 1.8129 2.1805 3.8659 0:00:00 1\n temperature limited to 5.000000e+03 in 178 cells\n1200 2.8107 1.8250 2.1862 3.8897 0:00:00 0\n')
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['final_iteration'],1200)
        self.assertTrue(result['recent_temperature_or_pressure_limiting'])

    def test_resolved_startup_limiting_and_small_residuals_pass_screen(self):
        result=self.assess('temperature limited to 5000 in 5 cells\niter continuity x-velocity y-velocity energy time/iter\n100 1e-5 1e-6 1e-6 1e-7 0:00:00 0\n')
        self.assertEqual(result['status'],'residual_screen_passed')

    def test_nonfinite_residual_fails_and_remains_json_serializable(self):
        result=self.assess('iter continuity x-velocity y-velocity energy time/iter\n100 nan 1e-6 1e-6 1e-7 0:00:00 0\n')
        self.assertEqual(result['status'],'failed')
        self.assertIsNone(result['final_residuals']['continuity'])

    def test_ambiguous_or_nonincreasing_native_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'history.out'
            path.write_text('"Iteration" "auto-wall-temp-min" "auto-wall-temp-avg" "auto-wall-temp-max"\n25 1000 2000 2500\n25 1100 2100 2600\n')
            with self.assertRaisesRegex(ValueError,'non-increasing'):parse_report_file(path)

class CompleteSolidTests(unittest.TestCase):
    def test_separate_solid_runner_and_comparison_use_current_temperature_sources(self):
        import test_adiabatic_study as fixtures
        from pipeline.config import load_config
        from pipeline.thermal import run_thermal_analysis
        from pipeline.temperature_export import export_temperatures, temperature_paths
        from pipeline.utils import file_sha256
        from pipeline.mechanical_reporting import mechanical_complete, create_mechanical_reports
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ta=yaml.safe_load((ROOT/'thermal_analysis.yaml').read_text())
            ta['loading']['heat_transfer_coefficient_w_m2k']=42
            ta['transient'].update(duration_s=2,initial_time_step_s=1,min_time_step_s=.1,max_time_step_s=2)
            thermal_path=root/'thermal.yaml';thermal_path.write_text(yaml.safe_dump(ta))
            for name in ('blunt','sharp'):
                work=root/'runs'/name;work.mkdir(parents=True)
                path=work/'_batch/config.yaml';path.parent.mkdir()
                path.write_text(yaml.safe_dump(fixtures.config(work)));cfg=load_config(path)
                for file in ('04_final.cas.h5','04_final.dat.h5'):(work/file).write_bytes(b'CFD fixture')
                source={key:file_sha256(work/file) for key,file in [('final_case','04_final.cas.h5'),('final_data','04_final.dat.h5')]}
                atomic_json(work/'summary.json',{'numerical_quality':{'status':'residual_screen_passed','source_sha256':source}})
                metadata=export_temperatures(fixtures.field_solver()[0],cfg,['body_wall'])
                metadata.update(source_sha256=source,wall_csv_sha256=file_sha256(temperature_paths(cfg)[0]))
                atomic_json(temperature_paths(cfg)[1],metadata)
                mapdl=SimpleNamespace(exit=lambda:None)
                def solve(engine,load,settings,size):
                    return np.arange(4)+1,np.c_[load.xy,np.zeros(4)], [np.array([400,500,600,700]),np.array([500,600,700,800])],[1,2],2
                with patch('ansys.mapdl.core.launch_mapdl',return_value=mapdl),patch('pipeline.thermal.solve_thermal',side_effect=solve):
                    result=run_thermal_analysis(cfd_config=path,thermal_config=thermal_path)
                self.assertEqual(result['backend'],'Mechanical APDL')
                self.assertTrue(mechanical_complete(work,cfg))
            rows,outputs=create_mechanical_reports(root/'runs',root/'runs/comparison/solid')
            self.assertEqual(len(rows),2)
            self.assertTrue(Path(outputs['report_manifest']).is_file())
            self.assertTrue((root/'runs/comparison/solid/wall_temperature_history_comparison.png').is_file())
            ta['loading']['heat_transfer_coefficient_w_m2k']=43
            thermal_path.write_text(yaml.safe_dump(ta))
            self.assertFalse(mechanical_complete(work,cfg))
            create_mechanical_reports(root/'runs',root/'runs/comparison/solid')
            self.assertFalse((root/'runs/comparison/solid').exists())


class NativeSchemaTests(unittest.TestCase):
    def test_native_solver_interfaces_match_pinned_fluent_251_schema(self):
        import inspect
        from ansys.fluent.core.generated.solver import settings_251 as native
        from ansys.mapdl.core._commands.solution.solid_surface_loads import SolidSurfaceLoads
        self.assertIn("courant_number",native.controls_1.child_names)
        self.assertIn("spatial_discretization",native.methods_1.child_names)
        self.assertEqual(native.compute_defaults.argument_names,["from_zone_type","from_zone_name","phase"])
        self.assertTrue({"line","lab","vali","valj","val2i","val2j"} <= set(inspect.signature(SolidSurfaceLoads.sfl).parameters))
