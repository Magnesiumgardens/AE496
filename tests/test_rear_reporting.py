"""Insulated rear faces stay out of surface plots without deleting solid data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.reporting import (create_comparison_outputs, read_csv, thermal_boundary_segments,
                                thermal_segment_flags, write_csv)


def scoped_case(work):
    (work / "_batch").mkdir(parents=True)
    (work / "_batch/config.yaml").write_text("reporting: {nose_direction: [1, 0]}\n")
    output = work / "thermal_analysis"; output.mkdir()
    # The active lower side wraps original index 3 -> 0. Original index 2
    # lies strictly inside the insulated back and is omitted from wall samples.
    xy = np.asarray([[2., 0.], [0., 1.], [0., 0.], [0., -1.]])
    flags = [1, 0, 0, 1]
    wall, flux, energy = [], [], []
    for time in (0., 5.):
        for index in (0, 1, 3):
            base = dict(time_s=time, surface_order_index=index, x_m=xy[index, 0], y_m=xy[index, 1],
                        thermal_segment_active=flags[index], wall_temperature_K=300. + time * (30 - index))
            quantities = dict(convective_heat_flux_into_solid_W_m2=100 + 10*index - 10*time,
                              radiative_heat_flux_out_W_m2=120 + 5*time,
                              net_heat_flux_into_solid_W_m2=-20 + 10*index - 15*time)
            cumulative = dict(cumulative_convective_energy_J_m2=(100 + 10*index)*time - 5*time**2,
                              cumulative_radiative_energy_J_m2=120*time + 2.5*time**2,
                              cumulative_net_energy_J_m2=(-20 + 10*index)*time - 7.5*time**2)
            wall.append(dict(base, **quantities, **cumulative))
            flux.append(dict(base, **quantities)); energy.append(dict(base, **cumulative))
    write_csv(output / "wall_temperature_history.csv", wall)
    write_csv(output / "surface_heat_flux_history.csv", flux)
    write_csv(output / "surface_energy_history.csv", energy)
    write_csv(output / "temperature_history.csv", [dict(time_s=0., minimum_K=300., mean_K=300., maximum_K=300.),
        dict(time_s=5., minimum_K=350., mean_K=460., maximum_K=600.)])
    summary = dict(status="complete", loading_mode="aerodynamic_surface_load",
                   settings={"rear_boundary": "insulated"}, visualization_settings={"make_gif": False})
    (output / "thermal_summary.json").write_text(json.dumps(summary))
    np.savez_compressed(output / "thermal_snapshots.npz", boundary_xy_m=xy,
        boundary_thermal_segment_active=np.asarray(flags),
        coordinates_m=np.c_[np.vstack([xy, [.5, 0.]]), np.zeros(5)],
        time_s=np.asarray([0., 5.]), temperatures_K=np.asarray([[300.]*5, [450., 400., 600., 350., 500.]]))
    return wall, flux, energy


def plotted_segments(ax):
    values = np.column_stack([ax.lines[0].get_xdata(), ax.lines[0].get_ydata()])
    # plot_thermal_boundary separates each saved segment with a NaN.
    return values.reshape(-1, 3, 2)[:, :2]


class RearReportingTests(unittest.TestCase):
    def test_mask_keeps_wraparound_forebody_edge_and_omits_rear_connection(self):
        records = [dict(surface_order_index=0, x_m=2., y_m=0., thermal_segment_active=1),
                   dict(surface_order_index=1, x_m=0., y_m=1., thermal_segment_active=0),
                   dict(surface_order_index=3, x_m=0., y_m=-1., thermal_segment_active=1)]
        actual = thermal_boundary_segments(records)
        np.testing.assert_equal(actual, [[[2., 0.], [0., 1.]], [[0., -1.], [2., 0.]]])
        legacy = [{key: value for key, value in row.items() if key != "thermal_segment_active"} for row in records]
        self.assertEqual(len(thermal_boundary_segments(legacy)), 3)
        self.assertEqual(len(thermal_boundary_segments(legacy, close_legacy=False)), 2)
        with self.assertRaisesRegex(ValueError, "missing"):
            thermal_segment_flags([records[0], legacy[1]])
        with self.assertRaisesRegex(ValueError, "zero or one"):
            thermal_segment_flags([dict(records[0], thermal_segment_active=.5)])

    def test_each_nose_and_comparison_have_scoped_flux_load_plots_and_full_solid_peak(self):
        from matplotlib.figure import Figure
        from pipeline import reporting
        expected = np.asarray([[[2., 0.], [0., 1.]], [[0., -1.], [2., 0.]]])
        map_segments, map_samples = [], []
        original_savefig = Figure.savefig
        def capture(fig, path, *args, **kwargs):
            if Path(path).name in ("final_surface_heat_flux_maps.png", "final_surface_heat_load_maps.png",
                                   "final_wall_heat_flux_map.png", "wall_boundary.png"):
                for ax in fig.axes:
                    if not ax.lines:continue  # shared colorbar axes
                    map_segments.append(plotted_segments(ax).copy())
                    map_samples.append(np.asarray(ax.collections[0].get_offsets()).copy())
            return original_savefig(fig, path, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "nose"
            wall, flux, energy = scoped_case(work)
            original_csv = {path: path.read_bytes() for path in (work / "thermal_analysis").glob("*.csv")}
            with patch.object(Figure, "savefig", new=capture):
                outputs = create_comparison_outputs(root, [], root / "comparison")
            self.assertTrue(map_segments)
            for segments, points in zip(map_segments, map_samples):
                np.testing.assert_equal(segments, expected)
                self.assertEqual(len(points), 3)
                self.assertFalse(np.any(np.all(points == [0., 0.], axis=1)))
            for target in (work / "reports", root / "comparison", root / "comparison/cases/nose"):
                for quantity in ("surface_heat_flux", "surface_heat_load"):
                    for suffix in ("_max_mean_history.png",):
                        self.assertTrue((target / (quantity + suffix)).is_file())
                    for suffix in ("_position.png", "_normalized_position.png", "_maps.png", "_profile.csv"):
                        self.assertTrue((target / ("final_" + quantity + suffix)).is_file())
            profiles = read_csv(outputs["final_surface_heat_flux_profile_csv"])
            self.assertEqual([int(row["surface_order_index"]) for row in profiles], [1, 0, 3])
            np.testing.assert_equal([float(row["net_heat_flux_into_solid_W_m2"]) for row in profiles], [-85., -95., -65.])
            self.assertTrue(all("thermal_segment_active" in row for row in profiles))
            raw_energy = read_csv(outputs["surface_energy_history_csv"])
            self.assertEqual(len(raw_energy), 6)
            self.assertNotIn(2, [int(row["surface_order_index"]) for row in raw_energy])
            # The insulated rear-only solid node is still the peak solid node.
            bars = read_csv(outputs["max_temperature_comparison_csv"])
            self.assertEqual(float(bars[0]["peak_temperature_K"]), 600.)
            self.assertTrue((work / "thermal_analysis/visualization/final_temperature_field.png").is_file())
            manifest = json.loads(Path(outputs["report_manifest"]).read_text())
            self.assertEqual(manifest["rear_insulated_cases"], ["nose"])
            self.assertIn("full solid", manifest["surface_load_plot_interpretation"])
            for path, data in original_csv.items():self.assertEqual(path.read_bytes(), data)


if __name__ == "__main__":unittest.main()
