"""Insulated rear cuts retain solid geometry but receive no external load."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from pipeline.aerothermal_loading import AerothermalLoad, integrated_powers, prepare_boundary_quadrature, read_load
from pipeline.reporting import read_csv, write_csv
from pipeline.thermal import apply_surface_loads, save_results, wall_rows
from pipeline.thermal_loading import ThermalAnalysisError
from test_aerothermal_thermal import settings


def masked_load():
    xy = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., .5]])
    h = np.array([100., 100., 100., 100., 0.])
    return AerothermalLoad(xy, h, h * 1500., np.full(5, 1000.), np.full(5, 300.),
                           h * 1200., h * 500., segment_active=np.array([1, 1, 1, 0, 0]))


def quadratic_mesh(load):
    xy = np.vstack([load.xy, .5 * (load.xy + np.roll(load.xy, -1, axis=0))])
    xyz = np.c_[xy, np.zeros(len(xy))]
    nnum = np.arange(1, len(xyz) + 1)
    class Engine:
        mesh = SimpleNamespace(nnum=nnum.copy())
        selected_lines = set()
        def lsel(self, mode, *args):
            if mode == "NONE":self.selected_lines = set()
            else:self.selected_lines.add(int(args[-1]) - 1)
        def nsll(self, *args):
            selected = set()
            for index in self.selected_lines:
                selected.update((index, (index + 1) % len(load.xy), len(load.xy) + index))
            self.mesh.nnum = nnum[sorted(selected)]
        def allsel(self):
            self.mesh.nnum = nnum.copy()
    engine = Engine()
    prepare_boundary_quadrature(engine, range(1, len(load.xy) + 1), load, nnum, xyz, .1)
    return engine, nnum, xyz


def write_load(path, load, clockwise=False):
    count = len(load.xy)
    indices = np.arange(count)[::-1] if clockwise else np.arange(count)
    outgoing = np.roll(load.segment_active, 1)[::-1] if clockwise else load.segment_active
    active = load.surface_active[indices]
    rows = [dict(surface_order_index=index, x_m=float(load.xy[old, 0]), y_m=float(load.xy[old, 1]),
                 aerodynamic_conductance_W_m2K=float(load.coefficient[old]),
                 aerodynamic_source_W_m2=float(load.aerodynamic_source[old]),
                 equilibrium_temperature_K=float(load.equilibrium_temperature[old]),
                 cold_wall_temperature_K=float(load.cold_temperature[old]),
                 cold_heat_flux_into_wall_W_m2=float(load.cold_flux[old]),
                 equilibrium_convective_heat_flux_W_m2=float(load.equilibrium_flux[old]),
                 driving_temperature_K=float(load.gas_temperature[old]),
                 thermal_active=int(active[index]), thermal_segment_active=int(outgoing[index]))
            for index, old in enumerate(indices)]
    write_csv(path, rows)
    return rows


class RearThermalTests(unittest.TestCase):
    def test_native_surface_loads_skip_all_rear_lines_and_retain_corner_heating(self):
        load = masked_load()
        _, _, validated = settings(200.)
        calls = []
        engine = SimpleNamespace(sfl=lambda *args: calls.append(args))
        apply_surface_loads(engine, [11, 12, 13, 14, 15], load, validated)
        self.assertEqual({call[0] for call in calls}, {11, 12, 13})
        self.assertEqual([call[1] for call in calls], ["CONV"] * 3)
        np.testing.assert_array_equal(load.surface_active, [True, True, True, True, False])
        self.assertEqual(load.gas_temperature[-1], 0.)

    def test_active_edge_quadrature_excludes_hot_rear_only_nodes_from_energy(self):
        load = masked_load()
        engine, nnum, xyz = quadratic_mesh(load)
        self.assertEqual(engine.selected_lines, {0, 1, 2})
        self.assertEqual({entry[3] for entry in load.boundary_quadrature}, {0, 1, 2})
        _, _, validated = settings(200.)
        values = np.full(len(nnum), 400.)
        values[[4, 8, 9]] = 10000.  # rear-only keypoint and rear midside nodes
        result = integrated_powers(load, values, validated)
        expected_convective = 3 * 100. * (1500. - 400.)
        expected_radiation = 3 * validated["emissivity"] * 5.670374419e-8 * (400.**4 - 3.**4)
        np.testing.assert_allclose(result, [expected_convective, expected_radiation, expected_convective - expected_radiation])

    def test_histories_and_flux_integrals_omit_rear_only_samples_but_keep_solid_peaks(self):
        load = masked_load()
        _, nnum, xyz = quadratic_mesh(load)
        _, _, validated = settings(200.)
        values = np.full(len(nnum), 500.)
        values[4] = 800.  # remains part of the solved solid and its peak metric
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            history, _, final, _ = save_results(output, load, nnum, xyz, [values], [200.], validated, .1)
            self.assertEqual(final["maximum_K"], 800.)
            self.assertEqual(history[-1]["maximum_K"], 800.)
            wall = read_csv(output / "wall_temperature_history.csv")
            self.assertEqual(len(wall), 8)
            self.assertEqual({int(row["surface_order_index"]) for row in wall}, {0, 1, 2, 3})
            self.assertEqual(len(read_csv(output / "wall_temperature_final.csv")), 4)
            self.assertEqual(len(read_csv(output / "thermal_nodes_final.csv")), len(nnum))
            for filename in ("surface_heat_flux_history.csv", "surface_energy_history.csv"):
                rows = read_csv(output / filename)
                self.assertEqual(len(rows), 8)
                self.assertEqual([int(row["thermal_segment_active"]) for row in rows[:4]], [1, 1, 1, 0])
            with np.load(output / "thermal_snapshots.npz") as snapshots:
                self.assertEqual(snapshots["boundary_xy_m"].shape, (5, 2))
                self.assertEqual(snapshots["temperatures_K"].shape, (2, len(nnum)))
                np.testing.assert_array_equal(snapshots["boundary_thermal_segment_active"], load.segment_active)

    def test_csv_orientation_keeps_outgoing_mask_on_the_same_physical_edges(self):
        load = masked_load()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "load.csv"
            for clockwise in (False, True):
                write_load(path, load, clockwise=clockwise)
                actual = read_load(path)
                np.testing.assert_allclose(actual.xy, load.xy)
                np.testing.assert_array_equal(actual.segment_active, load.segment_active)
                np.testing.assert_array_equal(actual.surface_active, load.surface_active)
                np.testing.assert_allclose(actual.gas_temperature, load.gas_temperature)
                scoped = read_load(path, scope=dict(exclude_rear_face=True, nose_direction=[1, 0],
                                                   rear_plane_tolerance_fraction=1e-7))
                np.testing.assert_array_equal(scoped.segment_active, load.segment_active)
                with self.assertRaisesRegex(ThermalAnalysisError, "configured geometric rear"):
                    read_load(path, scope=dict(exclude_rear_face=False))

    def test_inactive_zero_placeholders_do_not_allow_bad_active_conductance(self):
        load = masked_load()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "load.csv"
            rows = write_load(path, load)
            self.assertEqual(read_load(path).coefficient[-1], 0.)
            rows[0]["aerodynamic_conductance_W_m2K"] = -1.
            write_csv(path, rows)
            with self.assertRaisesRegex(ThermalAnalysisError, "positive"):
                read_load(path)
            rows = write_load(path, load)
            rows[-1]["thermal_active"] = 1
            write_csv(path, rows)
            with self.assertRaises(ThermalAnalysisError):read_load(path)


if __name__ == "__main__":unittest.main()
