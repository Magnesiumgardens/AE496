"""Geometry inventory and mk3/mk4 spatial comparisons from saved CSVs only."""
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pipeline.reporting import read_csv
from pipeline.utils import file_sha256


MK3 = [(0., 1.), (1., .2), (0., -1.)]
MK4 = [(0., 1.), (.5, .7), (1., .2), (.4, -.7), (0., -1.)]


def save_history(runs, name, points, *, broken_geometry=False):
    folder = runs / name / "thermal_analysis"
    folder.mkdir(parents=True)
    records = []
    for time in (0., 200.):
        for index, (x, y) in enumerate(points):
            # Different samples peak at different times. The envelope is
            # intentionally neither the initial nor final physical snapshot.
            cold = 1000. + 100. * index
            convection = cold if time == 0 else cold + (400. if index % 2 else -300.)
            radiation = 50. + index if time == 0 else 200. + index
            records.append(dict(time_s=time, surface_order_index=index,
                x_m=x, y_m=0. if broken_geometry else y,
                thermal_segment_active=0 if index == len(points) - 1 else 1,
                convective_heat_flux_into_solid_W_m2=convection,
                radiative_heat_flux_out_W_m2=radiation,
                net_heat_flux_into_solid_W_m2=convection-radiation))
    path = folder / "surface_heat_flux_history.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


def cad_files(folder, *names):
    folder.mkdir()
    for name in names:(folder / (name + ".x_t")).write_bytes(b"Inventory placeholder; do not read or mesh CAD")


class GeometryFluxComparisonTests(unittest.TestCase):
    def test_inventory_selects_mk3_mk4_instead_of_stale_run_folders(self):
        from pipeline.peak_flux_reporting import report_peak_heat_flux
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometries, runs = root / "geometry_batch", root / "runs"
            cad_files(geometries, "mk2", "mk3", "mk4")
            sources = [save_history(runs, "mk3", MK3), save_history(runs, "mk4", MK4),
                       save_history(runs, "mk2", MK3), save_history(runs, "old_unused", MK4)]
            before = {path: path.read_bytes() for path in [*sources, *geometries.iterdir()]}
            result = report_peak_heat_flux(runs, geometry_dir=geometries, cases=["mk3", "mk4"])
            self.assertEqual(result["case_count"], 2)
            self.assertEqual(len(result["groups"]), 1)
            self.assertEqual(set(result["groups"][0]["cases"]), {"mk3", "mk4"})
            inventory = {entry["case"]: entry for entry in result["geometry_inventory"]}
            for name in ("mk3", "mk4"):
                self.assertEqual(Path(inventory[name]["geometry_file"]).name, name + ".x_t")
                self.assertEqual(inventory[name]["run_case"], name)
            for path, contents in before.items():self.assertEqual(path.read_bytes(), contents)
            self.assertFalse((runs / "mk3/_batch").exists())
            self.assertFalse((runs / "mk3/02_mesh.msh.h5").exists())

    def test_pointwise_envelopes_preserve_each_noses_own_locations_and_peak_times(self):
        from pipeline.peak_flux_reporting import report_peak_heat_flux
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometries, runs = root / "geometry_batch", root / "runs"
            cad_files(geometries, "mk3", "mk4")
            save_history(runs, "mk3", MK3)
            save_history(runs, "mk4", MK4)
            result = report_peak_heat_flux(runs, geometry_dir=geometries)
            files = result["groups"][0]["files"]
            for stem in ("initial_heat_flux_spatial_comparison", "final_heat_flux_spatial_comparison",
                         "peak_heat_flux_spatial_envelope"):
                self.assertTrue(Path(files[stem + "_png"]).is_file())
                self.assertTrue(Path(files[stem + "_csv"]).is_file())
            envelope = read_csv(files["peak_heat_flux_spatial_envelope_csv"])
            for name, points in (("mk3", MK3), ("mk4", MK4)):
                rows = [row for row in envelope if row["design"] == name]
                self.assertEqual(len(rows), len(points))
                by_id = {int(row["surface_order_index"]): row for row in rows}
                self.assertEqual(set(by_id), set(range(len(points))))
                for index, point in enumerate(points):
                    row = by_id[index]
                    self.assertEqual((float(row["x_m"]), float(row["y_m"])), point)
                    expected_flux = 1000. + index * 100. + (400. if index % 2 else 0.)
                    self.assertEqual(float(row["peak_convective_heat_flux_W_m2"]), expected_flux)
                    self.assertEqual(float(row["time_of_peak_convective_flux_s"]), 200. if index % 2 else 0.)
                nose = max(range(len(points)), key=lambda index: points[index][0])
                self.assertAlmostEqual(float(by_id[0]["s_norm"]), 0.)
                self.assertAlmostEqual(float(by_id[nose]["s_norm"]), .5)
                self.assertAlmostEqual(float(by_id[len(points)-1]["s_norm"]), 1.)
                self.assertAlmostEqual(float(by_id[len(points)-1]["s_m"]),
                                       sum(math.dist(a, b) for a, b in zip(points, points[1:])))
            final = read_csv(files["final_heat_flux_spatial_comparison_csv"])
            self.assertEqual({float(row["time_s"]) for row in final}, {200.})
            self.assertEqual(len(final), len(MK3) + len(MK4))

    def test_casefold_matching_and_missing_selected_geometry_are_explicit(self):
        from pipeline.peak_flux_reporting import report_peak_heat_flux
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometries, runs = root / "geometry_batch", root / "runs"
            cad_files(geometries, "mk3", "mk4")
            save_history(runs, "MK3", MK3)
            result = report_peak_heat_flux(runs, geometry_dir=geometries, cases=["MK3", "Mk4"])
            self.assertEqual(result["case_count"], 1)
            self.assertEqual(result["groups"][0]["cases"], ["MK3"])
            self.assertEqual({row["case"].casefold() for row in result["excluded"]}, {"mk4"})
            self.assertTrue(result["excluded"][0]["reason"])
            self.assertFalse((runs / "mk4").exists())

    def test_unusable_spatial_coordinates_do_not_hide_available_scalar_peak_comparison(self):
        from pipeline.peak_flux_reporting import report_peak_heat_flux
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometries, runs = root / "geometry_batch", root / "runs"
            cad_files(geometries, "mk3")
            save_history(runs, "mk3", MK3, broken_geometry=True)
            result = report_peak_heat_flux(runs, geometry_dir=geometries)
            self.assertEqual(result["case_count"], 1)
            group = result["groups"][0]
            self.assertTrue(Path(group["files"]["peak_heat_flux_comparison_png"]).is_file())
            exclusions = group.get("spatial_excluded", []) or result.get("spatial_excluded", [])
            self.assertTrue(exclusions)
            self.assertEqual(exclusions[0]["case"], "mk3")

    def test_changed_CAD_under_the_same_name_cannot_reuse_an_older_saved_distribution(self):
        from pipeline.peak_flux_reporting import report_peak_heat_flux
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometries, runs = root / "geometry_batch", root / "runs"
            cad_files(geometries, "mk3")
            source = save_history(runs, "mk3", MK3)
            calibration = runs / "mk3/thermal_bc/surface_load_summary.json"
            calibration.parent.mkdir()
            calibration.write_text(json.dumps(dict(source_geometry_sha256=file_sha256(geometries / "mk3.x_t"))))
            receipt = source.parent / "thermal_summary.json"
            receipt.write_text(json.dumps(dict(status="complete", surface_package_sha256=file_sha256(calibration))))
            initial = report_peak_heat_flux(runs, geometry_dir=geometries)
            self.assertEqual(initial["case_count"], 1)
            retained = {path: path.read_bytes() for path in (source, calibration, receipt)}
            (geometries / "mk3.x_t").write_bytes(b"A revised mk3 CAD with the same filename")
            result = report_peak_heat_flux(runs, geometry_dir=geometries)
            self.assertEqual(result["case_count"], 0)
            self.assertEqual([row["case"] for row in result["excluded"]], ["mk3"])
            self.assertTrue(any(word in result["excluded"][0]["reason"].lower() for word in ("cad", "geometry")))
            for path, contents in retained.items():self.assertEqual(path.read_bytes(), contents)

    def test_unmatched_case_selector_is_reported_instead_of_silently_comparing_other_noses(self):
        from pipeline.peak_flux_reporting import report_peak_heat_flux
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometries, runs = root / "geometry_batch", root / "runs"
            cad_files(geometries, "mk3", "mk4")
            save_history(runs, "mk3", MK3)
            with self.assertRaisesRegex(ValueError, "[Mm]atch"):
                report_peak_heat_flux(runs, geometry_dir=geometries, cases=["mk9"])

    def test_cli_selects_geometry_batch_names_with_case_insensitive_patterns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometries, runs, output = root / "geometry_batch", root / "runs", root / "output"
            cad_files(geometries, "new_mk3", "new_mk4", "old_mk2")
            save_history(runs, "new_mk3", MK3)
            save_history(runs, "new_mk4", MK4)
            save_history(runs, "old_mk2", MK3)
            before = {path: path.read_bytes() for path in runs.rglob("*.csv")}
            environment = dict(os.environ, MPLCONFIGDIR=str(root / "matplotlib"), XDG_CACHE_HOME=str(root / "cache"))
            result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "study.py"),
                "report-heat-flux", "--runs-dir", str(runs), "--output-dir", str(output),
                "--geometry-dir", str(geometries), "--cases", "*MK3*", "*mk4*"],
                capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            table = read_csv(next(output.rglob("peak_heat_flux_comparison.csv")))
            self.assertEqual({row["design"] for row in table}, {"new_mk3", "new_mk4"})
            self.assertTrue(list(output.rglob("peak_heat_flux_spatial_envelope.png")))
            for path, contents in before.items():self.assertEqual(path.read_bytes(), contents)


if __name__ == "__main__":unittest.main()
