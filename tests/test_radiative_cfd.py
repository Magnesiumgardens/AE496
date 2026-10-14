"""Radiating-wall API readback and temperature-only single-solve export."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline.aero import wall_thermal_state, validate_saved_conditions
from pipeline.cfd import export_saved_case
from pipeline.solve import _set_wall_thermal, SolverPipelineError
from pipeline.temperature_export import export_temperatures, temperature_paths
from test_adiabatic_study import config, field_solver
from test_nose_trade import Reports


class NativeSetting:
    def __init__(self, value, allowed=None):
        self.value, self.allowed = value, allowed

    def get_state(self):
        return self.value

    def allowed_values(self):
        return self.allowed


class ConstantGroup:
    def __init__(self, value=0.0):
        self.option, self.value = "value", value

    def get_state(self):
        return {"option": self.option, "value": self.value}


class RadiatingThermal:
    def __init__(self, allowed=("Temperature", "Heat Flux", "rAdIaTiOn")):
        self.thermal_condition = NativeSetting("Temperature", list(allowed))
        self.external_emissivity = ConstantGroup()
        self.ext_rad_temperature = ConstantGroup()

    def __setattr__(self, name, value):
        if name == "thermal_condition" and name in self.__dict__:
            self.__dict__[name].value = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name in ("temperature", "heat_flux"):
            raise AssertionError("Radiative mode must not access prescribed temperature or zero heat flux")
        raise AttributeError(name)


def radiative_config(work):
    cfg = config(work)
    cfg["workflow"] = {"strategy": "single_solve"}
    cfg["thermal"] = dict(mode="radiative_equilibrium", emissivity=.9,
                          external_radiation_temperature_k=3., wall_temperature_k=300.)
    return cfg


class RadiativeCFDTests(unittest.TestCase):
    def test_native_radiation_enum_and_constant_groups_are_read_back(self):
        thermal = RadiatingThermal()
        requested = radiative_config(Path("unused"))["thermal"]
        result = _set_wall_thermal(SimpleNamespace(thermal=thermal), requested)
        self.assertEqual(result, {"thermal_condition": "rAdIaTiOn", "emissivity": .9,
                                  "external_radiation_temperature_k": 3.})
        self.assertEqual(thermal.external_emissivity.get_state(), {"option": "value", "value": .9})
        self.assertEqual(thermal.ext_rad_temperature.get_state(), {"option": "value", "value": 3.})
        from ansys.fluent.core.generated.solver import settings_251
        for name in ("external_emissivity", "ext_rad_temperature"):
            self.assertEqual(getattr(settings_251, name).child_names[:2], ["option", "value"])

    def test_missing_native_radiation_has_no_isothermal_or_adiabatic_fallback(self):
        thermal = RadiatingThermal(("Temperature", "Heat Flux"))
        with self.assertRaisesRegex(SolverPipelineError, "exposes"):
            _set_wall_thermal(SimpleNamespace(thermal=thermal), radiative_config(Path("unused"))["thermal"])
        self.assertEqual(thermal.thermal_condition.get_state(), "Temperature")
        self.assertEqual(thermal.external_emissivity.get_state()["value"], 0)

    def test_saved_radiation_readback_rejects_profiles_wrong_emissivity_and_other_modes(self):
        requested = radiative_config(Path("unused"))["thermal"]
        thermal = RadiatingThermal()
        _set_wall_thermal(SimpleNamespace(thermal=thermal), requested)
        thermal.external_emissivity.option = "profile"
        with self.assertRaisesRegex(ValueError, "constant"):
            wall_thermal_state(thermal, requested)
        thermal.external_emissivity.option = "value"
        thermal.external_emissivity.value = .8
        with self.assertRaisesRegex(ValueError, "external emissivity"):
            wall_thermal_state(thermal, requested)
        thermal.external_emissivity.value = .9
        for condition in ("Temperature", "Heat Flux"):
            thermal.thermal_condition = condition
            with self.assertRaisesRegex(ValueError, "radiative_equilibrium"):
                wall_thermal_state(thermal, requested)

    def test_saved_case_validates_wall_radiation_separately_from_volume_radiation(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = radiative_config(Path(directory))
            cfg["physics"] = {"viscous_model": "laminar", "air": {}}
            thermal = RadiatingThermal()
            _set_wall_thermal(SimpleNamespace(thermal=thermal), cfg["thermal"])
            far = SimpleNamespace(momentum=SimpleNamespace(
                mach_number=NativeSetting(8), gauge_pressure=NativeSetting(1090),
                flow_direction=NativeSetting([-1, 0])), thermal=SimpleNamespace(temperature=NativeSetting(226.5)))
            setup = SimpleNamespace(
                general=SimpleNamespace(operating_conditions=SimpleNamespace(operating_pressure=NativeSetting(0))),
                models=SimpleNamespace(viscous=SimpleNamespace(model=NativeSetting("laminar")),
                    energy=SimpleNamespace(enabled=NativeSetting(True)),
                    radiation=SimpleNamespace(model=NativeSetting("none"))),
                materials=SimpleNamespace(fluid={"air": SimpleNamespace(
                    density=SimpleNamespace(option=NativeSetting("ideal-gas")),
                    viscosity=SimpleNamespace(option=NativeSetting("sutherland")))}),
                boundary_conditions=SimpleNamespace(pressure_far_field=Reports(farfield=far),
                    wall=Reports(body_wall=SimpleNamespace(thermal=thermal))))
            solver = SimpleNamespace(settings=SimpleNamespace(setup=setup))
            validate_saved_conditions(solver, cfg)
            setup.models.radiation.model.value = "discrete-ordinates"
            with self.assertRaisesRegex(ValueError, "volumetric radiation"):
                validate_saved_conditions(solver, cfg)

    def test_export_temperature_field_not_flux_and_preserve_actual_wall_vertices(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = radiative_config(Path(directory))
            solver, requests = field_solver()
            result = export_temperatures(solver, cfg, ["body_wall"])
            self.assertEqual(result["thermal_mode"], "radiative_equilibrium")
            self.assertEqual(result["wall_temperature"]["length_weighted_mean_K"], 2150)
            self.assertEqual(result["near_wall_fluid_temperature"]["length_weighted_mean_K"], 2050)
            self.assertEqual(result["thermal_driver"], "wall_temperature_K")
            self.assertIn("Prescribed", result["solid_loading"])
            with Path(result["wall_nodes"]["csv"]).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertIn("wall_temperature_K", rows[0])
            self.assertNotIn("recovery_temperature_K", rows[0])
            self.assertEqual(sorted(float(row["wall_temperature_K"]) for row in rows), [220, 500, 1000, 2100])
            self.assertFalse(result["wall_matches_reference_temperature"])
            self.assertTrue(all(getattr(request, "field_name", "temperature") == "temperature"
                                for request in requests))
            summary = json.loads(temperature_paths(cfg)[1].read_text())
            self.assertEqual(summary["conditions"]["thermal"], cfg["thermal"])

    def test_reference_temperature_artifact_is_identified_without_relabeling_it_as_gas(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = radiative_config(Path(directory))
            solver, requests = field_solver()
            native = solver.fields.field_data.get_field_data
            def read(request):
                value = native(request)
                if getattr(request, "field_name", None) == "temperature" and request.boundary_value:
                    value[request.surfaces[0]][:] = 300.
                return value
            solver.fields.field_data.get_field_data = read
            result = export_temperatures(solver, cfg, ["body_wall"])
            self.assertTrue(result["wall_matches_reference_temperature"])
            self.assertEqual(result["wall_temperature"]["maximum_K"], 300)
            self.assertEqual(result["near_wall_fluid_temperature"]["maximum_K"], 2200)
            self.assertTrue(any("reference temperature" in message for message in result["limitations"]))

    def test_saved_export_requires_quality_and_creates_no_endpoint_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = radiative_config(root)
            for name in ("04_final.cas.h5", "04_final.dat.h5"):
                (root / name).write_bytes(b"single radiating solution")
            (root / "summary.json").write_text(json.dumps({"thermal_mode": "radiative_equilibrium"}))
            solver, requests = field_solver()
            solver.settings.file = SimpleNamespace(read_case_data=lambda **kwargs: None)
            solver.settings.setup.boundary_conditions.wall = Reports(body_wall=SimpleNamespace())
            solver.exit = lambda: None
            with patch("pipeline.cfd.launch_solver", return_value=solver), \
                 patch("pipeline.cfd.validate_saved_conditions"), \
                 patch("pipeline.solution_quality.require_usable_solution", return_value={"status": "not_verified"}):
                with self.assertRaisesRegex(ValueError, "passed residual screen"):
                    export_saved_case(cfg, aero=False)
            with patch("pipeline.cfd.launch_solver", return_value=solver), \
                 patch("pipeline.cfd.validate_saved_conditions"), \
                 patch("pipeline.solution_quality.require_usable_solution", return_value={"status": "residual_screen_passed"}):
                result = export_saved_case(cfg, aero=False)
            self.assertEqual(result["temperatures"]["thermal_mode"], "radiative_equilibrium")
            self.assertFalse((root / "cfd").exists())
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["wall_temperature_min_k"], 2000)
            self.assertEqual(summary["wall_temperature_avg_k"], 2150)
            self.assertEqual(summary["wall_temperature_max_k"], 2300)


if __name__ == "__main__":
    unittest.main()
