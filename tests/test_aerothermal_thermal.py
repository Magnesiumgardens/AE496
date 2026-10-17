"""Finite heating, implicit native BCs, and physical-time heat-energy reporting."""
import copy
import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from pipeline.aerothermal_loading import (AerothermalLoad, integrated_powers, prepare_boundary_quadrature,
                                          read_load, validate_settings, STEFAN_BOLTZMANN)
from pipeline.thermal import apply_surface_loads, save_results, solve_thermal
from pipeline.thermal_loading import ThermalAnalysisError


def settings(duration=400., emissivity=.9):
    cfd = {"thermal": {"mode": "radiative_equilibrium", "emissivity": emissivity,
                        "external_radiation_temperature_k": 3.}}
    ta = {"loading": {"mode": "aerodynamic_surface_load"}, "include_radiation": True,
          "material": {"conductivity_w_mk": 170., "density_kg_m3": 19250., "specific_heat_j_kgk": 130.},
          "initial_temperature_k": 300.,
          "transient": {"duration_s": duration, "initial_time_step_s": 1., "min_time_step_s": .1, "max_time_step_s": min(10., duration)}}
    return cfd, ta, validate_settings(cfd, ta)


def load_fixture(xy=None, h=100., driving=1500., emissivity=.9):
    xy = np.asarray(xy if xy is not None else [[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    count = len(xy)
    # An independently bracketed equilibrium temperature establishes the RE
    # anchor; no recovery-temperature assumption or prescribed T is needed.
    low, high = 300., driving
    for _ in range(80):
        middle = .5 * (low + high)
        balance = h * (driving - middle) - emissivity * STEFAN_BOLTZMANN * (middle**4 - 3.**4)
        if balance > 0: low = middle
        else: high = middle
    equilibrium = .5 * (low + high)
    return AerothermalLoad(xy, np.full(count, h), np.full(count, h * driving),
                           np.full(count, equilibrium), np.full(count, 300.),
                           np.full(count, h * (driving - 300.)), np.full(count, h * (driving - equilibrium)))


def quadratic_boundary(load):
    xy = np.vstack([load.xy, .5 * (load.xy + np.roll(load.xy, -1, axis=0))])
    xyz = np.c_[xy, np.zeros(len(xy))]
    nnum = np.arange(1, len(xyz) + 1)
    engine = SimpleNamespace(mesh=SimpleNamespace(nnum=nnum), lsel=lambda *args: None,
                             nsll=lambda *args: None, allsel=lambda: None)
    prepare_boundary_quadrature(engine, range(1, len(load.xy) + 1), load, nnum, xyz, .1)
    return nnum, xyz


class LoadPhysicsTests(unittest.TestCase):
    def test_calibrated_law_matches_both_wall_states_and_net_radiation_equilibrium(self):
        load = load_fixture()
        np.testing.assert_allclose(load.incoming_flux(load.cold_temperature), load.cold_flux)
        np.testing.assert_allclose(load.incoming_flux(load.equilibrium_temperature), load.equilibrium_flux)
        net = load.incoming_flux(load.equilibrium_temperature) - .9 * STEFAN_BOLTZMANN * (load.equilibrium_temperature**4 - 3.**4)
        np.testing.assert_allclose(net, 0., atol=1e-8)
        self.assertTrue((load.incoming_flux(np.full(4, 300.)) > load.incoming_flux(np.full(4, 600.))).all())

    def test_radiation_settings_must_match_calibration_and_cannot_be_disabled(self):
        cfd, ta, validated = settings()
        self.assertEqual(validated["trad"], 3.)
        self.assertEqual(validated["loading_mode"], "aerodynamic_surface_load")
        for changed in ({"include_radiation": False}, {"emissivity": .4}, {"radiation_temperature_k": 226.5}):
            candidate = {**ta, **changed}
            with self.assertRaises(ThermalAnalysisError): validate_settings(cfd, candidate)
        for duration in (200., 400.):
            candidate = copy.deepcopy(ta); candidate["transient"]["duration_s"] = duration
            self.assertEqual(validate_settings(cfd, candidate)["duration"], duration)

    def test_native_CONV_RAD_uses_positive_conductance_and_no_temperature_constraint(self):
        load = load_fixture(); _, _, validated = settings()
        calls = []
        engine = SimpleNamespace(sfl=lambda *args: calls.append(args),
                                 d=lambda *args: self.fail("Aerodynamic load must never constrain TEMP"))
        apply_surface_loads(engine, [11, 12, 13, 14], load, validated)
        self.assertEqual(len(calls), 8)
        self.assertEqual(calls[0], (11, "CONV", 100., 100., 1500., 1500.))
        self.assertEqual(calls[1], (11, "RAD", .9, .9, 3., 3.))
        # Native convention H*(Tbulk - Tsurface) adds heat into a colder solid.
        self.assertEqual(calls[0][2] * (calls[0][4] - 300.), 120000.)

    def test_node_csv_orientation_keeps_all_calibration_fields_attached(self):
        load = load_fixture()
        names = ("surface_order_index", "x_m", "y_m", "aerodynamic_conductance_W_m2K", "aerodynamic_source_W_m2",
                 "equilibrium_temperature_K", "cold_wall_temperature_K", "cold_heat_flux_into_wall_W_m2",
                 "equilibrium_convective_heat_flux_W_m2", "driving_temperature_K")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "load.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle); writer.writerow(names)
                for number, point in enumerate(load.xy[::-1]):
                    writer.writerow([number, *point, 100. + number, (100. + number) * 1500.,
                                     1000., 300., 120000., 50000., 1500.])
            read = read_load(path)
            self.assertEqual(dict(zip(map(tuple, read.xy), read.coefficient)),
                             {tuple(point): 100. + index for index, point in enumerate(load.xy[::-1])})

    def test_all_quadratic_edge_nodes_participate_in_radiation_integral(self):
        load = load_fixture(); _, _, validated = settings()
        _, xyz = quadratic_boundary(load)
        temperatures = np.r_[np.full(4, 400.), np.full(4, 700.)]
        power = integrated_powers(load, temperatures, validated)
        # Along each unit edge T(u)=400+1200*u-1200*u². Integrate T^4
        # analytically, independently of the production Gaussian quadrature.
        polynomial = np.polynomial.Polynomial([400., 1200., -1200.])
        radiation_integral = (polynomial**4).integ()
        expected_radiation = 4 * .9 * STEFAN_BOLTZMANN * (radiation_integral(1.) - radiation_integral(0.) - 3.**4)
        expected_convective = 4 * 100. * (1500. - polynomial.integ()(1.))
        np.testing.assert_allclose(power, [expected_convective, expected_radiation, expected_convective - expected_radiation], rtol=1e-12)
        # Sampling only CFD corners would miss all four hotter midside nodes.
        self.assertGreater(power[1], 4 * .9 * STEFAN_BOLTZMANN * (400.**4 - 3.**4))


class TransientResultsTests(unittest.TestCase):
    def test_uniform_lumped_limit_warms_gradually_and_reports_energy_from_initial_flux(self):
        load = load_fixture(); nnum, xyz = quadratic_boundary(load)
        _, _, validated = settings(duration=400.)
        capacity_per_wetted_area = 100000.
        h, driving = 100., 1500.
        times = np.linspace(.2, 400., 2000)
        # Independent reduced physical model: a highly conducting uniform body.
        # This checks the balance and saved energy, without pretending it is a
        # substitute for a licensed MAPDL native-engine convergence test.
        def balance(time, state):
            return [(h * (driving - state[0]) - .9 * STEFAN_BOLTZMANN * (state[0]**4 - 3.**4)) / capacity_per_wetted_area]
        solution = solve_ivp(balance, [0., 400.], [300.], t_eval=times, rtol=1e-11, atol=1e-10)
        temperatures = [np.full(len(nnum), value) for value in solution.y[0]]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _, _, final, heat = save_results(output, load, nnum, xyz, temperatures, times, validated, .1)
            with (output / "wall_temperature_history.csv").open() as handle:
                wall = list(csv.DictReader(handle))
            self.assertFalse(any("prescribed" in key for key in wall[0]))
            self.assertEqual(float(wall[0]["wall_temperature_K"]), 300.)
            self.assertLess(float(wall[4]["wall_temperature_K"]), 301.)
            self.assertLess(final["maximum_K"], load.equilibrium_temperature[0])
            initial_net = h * (driving - 300.) - .9 * STEFAN_BOLTZMANN * (300.**4 - 3.**4)
            self.assertAlmostEqual(float(wall[0]["net_heat_flux_into_solid_W_m2"]), initial_net)
            self.assertEqual(float(wall[0]["cumulative_net_energy_J_m2"]), 0.)
            expected_heat = 4 * capacity_per_wetted_area * (solution.y[0, -1] - 300.)
            self.assertAlmostEqual(heat / expected_heat, 1., places=6)
            with (output / "surface_energy_history.csv").open() as handle:
                energies = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(energies[-1]["cumulative_net_energy_J_m2"]), heat / 4, places=6)
            for name in ("surface_heat_flux_history.csv", "global_energy_history.csv"):
                self.assertTrue((output / name).is_file())
            with np.load(output / "thermal_snapshots.npz") as saved:
                self.assertIn("boundary_aerodynamic_source_W_m2", saved)
                self.assertNotIn("boundary_prescribed_temperature_K", saved)

    def test_native_transient_configuration_starts_uniform_and_never_issues_D_TEMP(self):
        load = load_fixture([[0., 0.], [1., 0.], [0., 1.]])
        nnum, xyz = quadratic_boundary(load)
        _, _, validated = settings(duration=200.)
        calls = []
        class Engine:
            mesh = SimpleNamespace(nnum=nnum, nodes=xyz, enum=np.arange(1, 2))
            post_processing = SimpleNamespace(nsets=2, time_values=[2., 200.], nodal_temperature=lambda: np.full(6, 500.))
            def __getattr__(self, name):
                def command(*args, **kwargs):
                    calls.append((name, args, kwargs))
                    if name == "d": raise AssertionError("No TEMP constraint allowed")
                    if name in {"k", "l"}: return sum(call[0] == name for call in calls)
                    if name == "al": return 1
                    return ""
                return command
        result = solve_thermal(Engine(), load, validated, .1)
        self.assertEqual(result[3], [2., 200.])
        self.assertIn(("tunif", (300.,), {}), calls)
        self.assertIn(("time", (200.,), {}), calls)
        self.assertIn(("deltim", (1., .1, 10.), {}), calls)
        self.assertTrue(any(name == "sfl" and args[1] == "CONV" for name, args, kwargs in calls))
        self.assertTrue(any(name == "sfl" and args[1] == "RAD" for name, args, kwargs in calls))
        self.assertFalse(any(name == "d" for name, args, kwargs in calls))


if __name__ == "__main__": unittest.main()
