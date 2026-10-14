"""Regression for Fluent251 creatable dictionaries with no initial entries."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import yaml

from pipeline.aero import reference_state
from pipeline.solver_numerics import configure_numerics, initialize_flow


class NativeDictionary:
    def __init__(self):self.values={}
    def get_object_names(self):return list(self.values)
    def __setitem__(self,key,value):self.values[key]=value
    def __getitem__(self,key):
        if key not in self.values:raise KeyError(key)
        return SimpleNamespace(get_state=lambda:self.values[key])


class SolverNumericsTests(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'config.yaml').read_text())
        self.schemes=NativeDictionary()
        self.defaults=NativeDictionary()
        self.calls=[]
        initialization=SimpleNamespace(defaults=self.defaults,
            compute_defaults=lambda **kwargs:self.calls.append(kwargs),
            standard_initialize=lambda:self.calls.append('standard'),
            hybrid_initialize=lambda:self.calls.append('hybrid'))
        self.solution=SimpleNamespace(controls=SimpleNamespace(),initialization=initialization,
            methods=SimpleNamespace(spatial_discretization=SimpleNamespace(discretization_scheme=self.schemes)))
        self.solver=SimpleNamespace(settings=SimpleNamespace(solution=self.solution))

    def test_missing_flow_entry_is_created_then_read_back(self):
        self.assertEqual(self.schemes.get_object_names(),[])
        result=configure_numerics(self.solver,self.cfg)
        self.assertEqual(result['flow_scheme'],'first-order-upwind')
        self.assertEqual(self.schemes.values,{'flow':'first-order-upwind'})
        self.assertEqual(self.solution.controls.courant_number,.1)

    def test_initialization_defaults_need_not_be_precreated(self):
        self.cfg['physics']['operating_pressure_pa']=100
        result=initialize_flow(self.solver,self.cfg,'farfield')
        self.assertEqual(self.calls,[{'from_zone_type':'pressure-far-field','from_zone_name':'farfield'},'standard'])
        self.assertEqual(self.defaults.values['pressure'],990)
        self.assertEqual(self.defaults.values['temperature'],226.5)
        self.assertAlmostEqual(self.defaults.values['x-velocity'],-reference_state(self.cfg)['velocity_m_s'])
        self.assertEqual(result['method'],'freestream')

    def test_rejected_native_scheme_is_not_silently_skipped(self):
        class RejectedDictionary(NativeDictionary):
            def __setitem__(self,key,value):raise RuntimeError('Native scheme not supported')
        self.solution.methods.spatial_discretization.discretization_scheme=RejectedDictionary()
        with self.assertRaisesRegex(ValueError,'native entries=\\[\\].*not supported'):
            configure_numerics(self.solver,self.cfg)

    def test_pinned_native_types_support_creation_on_assignment(self):
        from ansys.fluent.core.generated.solver import settings_251
        from ansys.fluent.core.solver.flobject import CreatableNamedObjectMixin
        self.assertTrue(issubclass(settings_251.discretization_scheme,CreatableNamedObjectMixin))
        self.assertTrue(issubclass(settings_251.defaults,CreatableNamedObjectMixin))
