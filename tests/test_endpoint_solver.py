"""Native boundary groups and shared solver warm-start behavior."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from pipeline.endpoint_workflow import effective_cfg
from pipeline.solve import _set_wall_thermal, solve_case
from pipeline.solver_numerics import configure_numerics
from test_solver_numerics import ServerDictionary
from pipeline.temperature_export import temperature_conditions
from pipeline.utils import atomic_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]


class Collection(dict):
    def get_object_names(self):return list(self)


class Direction:
    def __init__(self):self.value=[-1.,0.]
    def get_state(self):return self.value
    def __setitem__(self,index,value):self.value[index]=value


def scalar(value):return SimpleNamespace(get_state=lambda:value)


class EndpointSolverTests(unittest.TestCase):
    def test_original_native_numerics_are_retained_and_recorded(self):
        schemes=ServerDictionary({'amg-c':'second-order-upwind'}, {'amg-c':['first-order-upwind','second-order-upwind']})
        controls=SimpleNamespace(courant_number=scalar(5.))
        solver=SimpleNamespace(settings=SimpleNamespace(solution=SimpleNamespace(controls=controls,
            methods=SimpleNamespace(spatial_discretization=SimpleNamespace(discretization_scheme=schemes)))))
        cfg={'solver':{'courant_number':None,'flow_scheme':None}}
        result=configure_numerics(solver,cfg)
        self.assertEqual((result['courant_number'],result['flow_scheme']),(5.,'second-order-upwind'))
        self.assertEqual((result['courant_source'],result['flow_scheme_source']),('native','native'))
        self.assertEqual(controls.courant_number.get_state(),5.)
        self.assertEqual(schemes.values,{'amg-c':'second-order-upwind'})
        self.assertEqual(schemes.created,[])

    def test_cold_wall_uses_the_native_constant_temperature_group(self):
        temperature = SimpleNamespace()
        thermal = SimpleNamespace(temperature=temperature)
        result = _set_wall_thermal(SimpleNamespace(thermal=thermal),dict(mode='fixed_temperature',wall_temperature_k=300))
        self.assertEqual(thermal.thermal_condition,'Temperature')
        self.assertEqual((temperature.option,temperature.value),('value',300))
        self.assertEqual(result['wall_temperature_k'],300)

    def test_warm_reference_reads_case_data_switches_wall_and_never_initializes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = yaml.safe_load((ROOT/'tests/fixtures/two_endpoint_config.yaml').read_text())
            cfg['project']['work_dir']=str(root)
            cfg['geometry']['input_file']=str(root/'nose.x_t')
            cfg['solver']['temperature_history']['enabled']=False
            (root/'nose.x_t').write_text('CAD fixture'); (root/'02_mesh.msh.h5').write_bytes(b'mesh')
            cold, adiabatic = effective_cfg(cfg,'cold_wall'), effective_cfg(cfg,'adiabatic')
            cold_work=Path(cold['project']['work_dir']); cold_work.mkdir(parents=True)
            for name in ('04_final.cas.h5','04_final.dat.h5'):(cold_work/name).write_bytes(name.encode())
            source={key:file_sha256(cold_work/name) for key,name in
                (('final_case','04_final.cas.h5'),('final_data','04_final.dat.h5'))}
            atomic_json(cold_work/'summary.json',dict(thermal_mode='fixed_temperature',source_conditions=temperature_conditions(cold),
                source_mesh_sha256=file_sha256(root/'02_mesh.msh.h5'),source_geometry_sha256=file_sha256(root/'nose.x_t'),
                numerical_quality=dict(status='residual_screen_passed',source_sha256=source)))
            calls=[]
            thermal=SimpleNamespace(thermal_condition=scalar('Temperature'),temperature=scalar({'option':'value','value':300}),heat_flux=SimpleNamespace())
            far=SimpleNamespace(momentum=SimpleNamespace(gauge_pressure=scalar(1090),mach_number=scalar(8),flow_direction=Direction()),
                thermal=SimpleNamespace(temperature=scalar(226.5)))
            sutherland=SimpleNamespace(option=scalar('three-coefficient-method'),
                reference_viscosity=scalar(1.716e-5),reference_temperature=scalar(273.11),effective_temperature=scalar(110.56))
            setup=SimpleNamespace(general=SimpleNamespace(solver=SimpleNamespace(),
                operating_conditions=SimpleNamespace(operating_pressure=scalar(0))),
                models=SimpleNamespace(energy=SimpleNamespace(enabled=scalar(True),viscous_dissipation=SimpleNamespace(is_active=lambda:False)),
                    radiation=SimpleNamespace(model=scalar('none')),viscous=SimpleNamespace(model=scalar('laminar'))),
                materials=SimpleNamespace(fluid={'air':SimpleNamespace(density=SimpleNamespace(option=scalar('ideal-gas')),
                    viscosity=SimpleNamespace(option=scalar('sutherland'),sutherland=sutherland))}),
                boundary_conditions=SimpleNamespace(wall=Collection(body_wall=SimpleNamespace(thermal=thermal)),
                    pressure_far_field=Collection(farfield=far),interior=Collection()))
            def read_mesh(**kwargs):raise AssertionError('Warm reference must retain the loaded solution')
            def write_pair(file_name):
                path=Path(file_name); path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'adiabatic case')
                path.with_name('04_final.dat.h5').write_bytes(b'adiabatic data')
            solver=SimpleNamespace(settings=SimpleNamespace(setup=setup,
                file=SimpleNamespace(read_mesh=read_mesh,read_case_data=lambda **kwargs:calls.append(('read',kwargs)),
                    write=lambda **kwargs:None,write_case_data=write_pair),
                solution=SimpleNamespace(run_calculation=SimpleNamespace(iterate=lambda **kwargs:calls.append(('iterate',kwargs))))),
                tui=SimpleNamespace(mesh=SimpleNamespace(check=lambda:calls.append(('mesh_check',{})))),exit=lambda:calls.append(('exit',{})))
            with patch('pipeline.solve.launch_solver',return_value=solver), \
                 patch('pipeline.solver_numerics.configure_numerics',return_value={'source':'native'}), \
                 patch('pipeline.solver_numerics.configure_residual_criteria',return_value={'energy':1e-3}), \
                 patch('pipeline.solver_numerics.initialize_flow',side_effect=AssertionError('No initialization allowed')):
                result=solve_case(adiabatic,root/'02_mesh.msh.h5',reference_cfg=cold)
            self.assertEqual(calls[0],('read',{'file_name':str(cold_work/'04_final.cas.h5')}))
            self.assertEqual([args['iter_count'] for name,args in calls if name=='iterate'],[600])
            self.assertEqual(thermal.thermal_condition,'Heat Flux')
            self.assertEqual((thermal.heat_flux.option,thermal.heat_flux.value),('value',0.))
            self.assertEqual(result['reference_source_sha256'],source)
            self.assertFalse(result['initialization']['initialization_performed'])
            self.assertEqual(result['thermal_mode'],'adiabatic')
            self.assertEqual(calls[-1][0],'exit')


if __name__ == '__main__':unittest.main()
