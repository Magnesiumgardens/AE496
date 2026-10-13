"""Numerical/load/output checks; Ansys installation is not required."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline.thermal import (
    STEFAN_BOLTZMANN, apply_surface_loads, config_paths, save_results,
    temperature_stats, wall_rows,
)
from pipeline.thermal_loading import (
    ThermalAnalysisError, ThermalLoad, integrate_boundary, read_boundary,
    validate_settings,
)


def settings():
    return {"duration": 10, "initial_t": 300, "include_rad": False,
            "emissivity": 0.9, "trad": 3}


class ThermalLoadingTests(unittest.TestCase):
    xy = np.asarray([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])

    def test_directflux_and_radiation_have_opposite_signs(self):
        load = ThermalLoad(self.xy, np.full(4, 100.))
        cfg = {**settings(), "include_rad": True}
        values = np.asarray([300., 400., 500., 600.])
        rows = wall_rows(5, load, values, np.arange(1, 5), np.arange(4), np.zeros(4), cfg)
        for row, value in zip(rows, values):
            expected = 0.9 * STEFAN_BOLTZMANN * (value**4 - 3**4)
            self.assertAlmostEqual(row["radiative_heat_flux_out_W_m2"], expected)
            self.assertAlmostEqual(row["net_surface_heat_flux_into_wall_W_m2"], 100 - expected)
        np.testing.assert_equal(load.incoming_flux(values), [100] * 4)

    def test_mapdl_loads_preserve_variable_spatial_flux_and_close_polygon(self):
        class Recorder:
            calls = []
            def sfl(self, *args):
                self.calls.append(args)
        mapdl = Recorder()
        apply_surface_loads(mapdl, [1, 2, 3, 4], ThermalLoad(self.xy, np.array([10., 20., 30., 40.])), settings())
        self.assertEqual(mapdl.calls[0], (1, "HFLUX", 10., 20.))
        self.assertEqual(mapdl.calls[-1], (4, "HFLUX", 40., 10.))
        self.assertEqual(integrate_boundary(self.xy, np.full(4, 100.)), 400)

    def test_clockwise_csv_reverses_geometry_and_flux_together(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bc.csv"
            path.write_text("surface_order_index,x_m,y_m,heat_flux_into_wall_W_m2\n0,0,0,1\n1,0,1,2\n2,1,1,3\n3,1,0,4\n")
            xy, q = read_boundary(path)
            actual = {tuple(point): value for point, value in zip(xy, q)}
            self.assertEqual(actual, {(0., 0.): 1., (0., 1.): 2., (1., 1.): 3., (1., 0.): 4.})
            area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
            self.assertGreater(area, 0)

    def test_legacy_loading_modes_and_zero_timestep_are_rejected(self):
        for mode in ("auto", "convective_from_cfd"):
            with self.assertRaises(ThermalAnalysisError):
                validate_settings({}, {"loading": {"mode": mode}})
        ta = {"material": {"conductivity_w_mk": 170, "density_kg_m3": 19250, "specific_heat_j_kgk": 130},
              "transient": {"duration_s": 10, "initial_time_step_s": 0}}
        with self.assertRaises(ThermalAnalysisError):
            validate_settings({}, ta)

    def test_selected_config_controls_case_and_relative_boundary_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {"project": {"work_dir": str(root / "case")}}
            work, boundary, output = config_paths(cfg, {}, root / "settings.yaml")
            self.assertEqual(boundary, work / "thermal_bc" / "body_wall_faces.csv")
            self.assertEqual(output, work / "thermal_analysis")
            _, override, _ = config_paths(cfg, {"boundary_csv": "loads/body.csv"}, root / "settings.yaml")
            self.assertEqual(override, root / "loads" / "body.csv")
            with self.assertRaises(ThermalAnalysisError):
                config_paths(cfg, {}, root / "settings.yaml", root / "different_case")


class SavedThermalResultsTests(unittest.TestCase):
    xy = ThermalLoadingTests.xy
    xyz = np.column_stack([xy, np.zeros(4)])

    def test_full_histories_initial_state_final_data_and_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            load = ThermalLoad(self.xy, np.full(4, 100.))
            history, snapshots, final, net_load = save_results(root, load, [11, 12, 13, 14], self.xyz,
                [np.array([310, 320, 330, 340]), np.array([350, 360, 370, 380])], [5, 10], settings(), 0.1)
            self.assertEqual(history[0], {"time_s": 0., "minimum_K": 300., "mean_K": 300., "maximum_K": 300.})
            self.assertEqual(final["maximum_K"], 380)
            self.assertEqual(net_load, 4000.)
            with (root / "wall_temperature_history.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual(float(rows[-1]["wall_temperature_K"]), 380)
            self.assertEqual(int(rows[-1]["nearest_thermal_node_id"]), 14)
            with np.load(root / "thermal_snapshots.npz") as archive:
                np.testing.assert_equal(archive["time_s"], [0, 5, 10])
                np.testing.assert_equal(archive["temperatures_K"][-1], [350, 360, 370, 380])
            self.assertTrue((root / "thermal_nodes_final.csv").is_file())
            self.assertTrue((root / "wall_temperature_final.csv").is_file())
            self.assertEqual(snapshots[-1]["time_s"], 10.)

    def test_incomplete_times_and_bad_node_mapping_fail_before_saving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            load = ThermalLoad(self.xy, np.ones(4))
            for times in ([5], [10, 5], [5, float("nan")]):
                with self.assertRaises(ThermalAnalysisError):
                    save_results(root, load, [1, 2, 3, 4], self.xyz, [np.full(4, 350)] * len(times), times, settings(), 0.1)
            with self.assertRaises(ThermalAnalysisError):
                save_results(root, load, [1, 2, 3, 4], self.xyz + [0.5, 0, 0], [np.full(4, 350)], [10], settings(), 0.1)
            self.assertFalse((root / "wall_temperature_history.csv").exists())

    def test_nonphysical_results_are_not_silently_cleaned(self):
        for values in ([300, 0], [300, -50], [300, float("nan")]):
            with self.assertRaises(ThermalAnalysisError):
                temperature_stats(values)


if __name__ == "__main__":
    unittest.main()
