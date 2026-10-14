"""Use Fluent server dictionaries which forbid creating equation entries."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import yaml

from pipeline.aero import reference_state
from pipeline.solver_numerics import configure_numerics, initialize_flow


class ServerDictionary:
    def __init__(self,values,choices=None):
        self.values=dict(values)
        self.choices=choices or {}
        self.created=[]
    def get_object_names(self):return list(self.values)
    def __setitem__(self,key,value):
        if key not in self.values:
            self.created.append(key)
            raise RuntimeError('api-create-obj: new objects of this type cannot be created')
        self.values[key]=value
    def __getitem__(self,key):
        if key not in self.values:raise KeyError(key)
        return SimpleNamespace(get_state=lambda:self.values[key],allowed_values=lambda:self.choices.get(key,[]))


class SolverNumericsTests(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'config.yaml').read_text())
        self.schemes=ServerDictionary({'amg-c':'second-order-upwind'},
                                     {'amg-c':['first-order-upwind','second-order-upwind']})
        self.defaults=ServerDictionary({'pressure':0.,'temperature':300.,'x-velocity':0.,'y-velocity':0.})
        self.calls=[]
        initialization=SimpleNamespace(defaults=self.defaults,
            compute_defaults=lambda **kwargs:self.calls.append(kwargs),
            standard_initialize=lambda:self.calls.append('standard'),
            hybrid_initialize=lambda:self.calls.append('hybrid'))
        self.solution=SimpleNamespace(controls=SimpleNamespace(),initialization=initialization,
            methods=SimpleNamespace(spatial_discretization=SimpleNamespace(discretization_scheme=self.schemes)))
        self.solver=SimpleNamespace(settings=SimpleNamespace(solution=self.solution))

    def test_real_amg_c_entry_is_used_without_creating_flow(self):
        result=configure_numerics(self.solver,self.cfg)
        self.assertEqual(result['flow_scheme'],'first-order-upwind')
        self.assertEqual(result['flow_scheme_entry'],'amg-c')
        self.assertEqual(self.schemes.values,{'amg-c':'first-order-upwind'})
        self.assertEqual(self.schemes.created,[])
        self.assertEqual(self.solution.controls.courant_number,.1)

    def test_flow_alias_is_supported_only_when_exposed_by_server(self):
        self.schemes.values={'flow':'second-order-upwind'}
        self.schemes.choices={'flow':['first-order-upwind','second-order-upwind']}
        self.assertEqual(configure_numerics(self.solver,self.cfg)['flow_scheme_entry'],'flow')
        self.assertEqual(self.schemes.created,[])

    def test_unavailable_scheme_has_native_choices_and_does_not_change_it(self):
        self.cfg['solver']['flow_scheme']='unsupported'
        with self.assertRaisesRegex(ValueError,"native choices=.*first-order-upwind"):
            configure_numerics(self.solver,self.cfg)
        self.assertEqual(self.schemes.values['amg-c'],'second-order-upwind')

    def test_unrecognized_equation_is_not_guessed_or_created(self):
        self.schemes.values={'temperature':'first-order-upwind'}
        with self.assertRaisesRegex(ValueError,"native entries=.*temperature"):
            configure_numerics(self.solver,self.cfg)
        self.assertEqual(self.schemes.created,[])

    def test_freestream_defaults_use_gauge_pressure_and_negative_x_velocity(self):
        self.cfg['physics']['operating_pressure_pa']=100
        result=initialize_flow(self.solver,self.cfg,'farfield')
        self.assertEqual(self.calls,[{'from_zone_type':'pressure-far-field','from_zone_name':'farfield'},'standard'])
        self.assertEqual(self.defaults.values['pressure'],990)
        self.assertEqual(self.defaults.values['temperature'],226.5)
        self.assertAlmostEqual(self.defaults.values['x-velocity'],-reference_state(self.cfg)['velocity_m_s'])
        self.assertEqual(result['method'],'freestream')
        self.assertEqual(self.defaults.created,[])

    def test_missing_initialization_defaults_report_names_without_creation(self):
        self.defaults.values.pop('x-velocity')
        with self.assertRaisesRegex(ValueError,'native defaults='):
            initialize_flow(self.solver,self.cfg,'farfield')
        self.assertEqual(self.defaults.created,[])
        self.assertNotIn('standard',self.calls)
