"""Report saved heat-flux peaks without solver files or native Ansys clients."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline.reporting import read_csv
from pipeline.utils import file_sha256


def sample(time, index, convection, radiation, outgoing=1):
    return dict(time_s=time, surface_order_index=index, x_m=index / 10., y_m=index / 20.,
                thermal_segment_active=outgoing,
                convective_heat_flux_into_solid_W_m2=convection,
                radiative_heat_flux_out_W_m2=radiation,
                net_heat_flux_into_solid_W_m2=convection-radiation)


def save_case(root, name, samples, filename="surface_heat_flux_history.csv", summary=None):
    output = root / name / "thermal_analysis"
    output.mkdir(parents=True)
    source = output / filename
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    if summary is not None:
        (output / "thermal_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return source


def report(root):
    from pipeline.peak_flux_reporting import report_peak_heat_flux
    return report_peak_heat_flux(root)


class PeakFluxReportingTests(unittest.TestCase):
    def test_archived_csvs_compare_signed_spatial_and_time_peaks_without_native_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The shared corner with outgoing segment_active=0 still belongs
            # to its incoming heated edge and must remain in the comparison.
            blunt = save_case(root, "blunt", [sample(0, 4, 900, 100),
                sample(0, 5, 1000, 300, outgoing=0), sample(200, 4, 200, 400)])
            sharp = save_case(root, "sharp", [sample(0, 7, 100, 1000), sample(200, 7, 60, 100)])
            original = {path: path.read_bytes() for path in (blunt, sharp)}
            result = report(root)
            self.assertEqual(result["case_count"], 2)
            self.assertEqual(len(result["groups"]), 1)
            self.assertEqual(result["excluded"], [])
            group = result["groups"][0]
            rows = {row["design"]: row for row in read_csv(group["files"]["peak_heat_flux_comparison_csv"])}
            self.assertEqual(set(rows), {"blunt", "sharp"})
            self.assertEqual(float(rows["blunt"]["peak_convective_heat_flux_W_m2"]), 1000.)
            self.assertEqual(float(rows["blunt"]["time_of_peak_convective_flux_s"]), 0.)
            self.assertEqual(int(rows["blunt"]["peak_convective_surface_order_index"]), 5)
            self.assertEqual(float(rows["blunt"]["peak_convective_x_m"]), .5)
            self.assertEqual(float(rows["blunt"]["peak_convective_y_m"]), .25)
            self.assertEqual(float(rows["blunt"]["peak_net_heat_flux_W_m2"]), 800.)
            self.assertEqual(float(rows["blunt"]["peak_radiative_heat_flux_W_m2"]), 400.)
            self.assertEqual(float(rows["sharp"]["peak_net_heat_flux_W_m2"]), -40.)
            self.assertEqual(rows["blunt"]["validation_status"], "saved_csv_only")
            self.assertEqual(Path(rows["blunt"]["source_csv"]).resolve(), blunt.resolve())
            self.assertTrue(Path(group["files"]["peak_heat_flux_comparison_png"]).is_file())
            self.assertTrue(Path(result["manifest"]).is_file())
            for path, contents in original.items():self.assertEqual(path.read_bytes(), contents)
            self.assertFalse((root / "blunt/_batch").exists())
            self.assertFalse((root / "blunt/04_final.cas.h5").exists())

    def test_schema_compatible_wall_history_fallback_does_not_infer_flux_from_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = save_case(root, "with_flux", [sample(0, 1, 500, 10), sample(200, 1, 300, 50)],
                               filename="wall_temperature_history.csv")
            save_case(root, "temperature_only", [dict(time_s=0, surface_order_index=0, x_m=0., y_m=0.,
                wall_temperature_K=300.), dict(time_s=200, surface_order_index=0, x_m=0., y_m=0.,
                wall_temperature_K=1800.)], filename="wall_temperature_history.csv")
            result = report(root)
            self.assertEqual(result["case_count"], 1)
            rows = read_csv(result["groups"][0]["files"]["peak_heat_flux_comparison_csv"])
            self.assertEqual(Path(rows[0]["source_csv"]).resolve(), source.resolve())
            self.assertEqual({row["case"] for row in result["excluded"]}, {"temperature_only"})
            self.assertTrue(result["excluded"][0]["reason"])

    def test_explicit_failed_energy_and_hash_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = [sample(0, 1, 500, 10), sample(200, 1, 300, 50)]
            save_case(root, "failed_energy", values, summary=dict(status="complete", energy_balance=dict(status="failed")))
            save_case(root, "hash_mismatch", values, summary=dict(status="complete", energy_balance=dict(status="passed"),
                result_sha256={"surface_heat_flux_history.csv": "not-the-saved-file-hash"}))
            save_case(root, "failed_run", values, summary=dict(status="failed"))
            save_case(root, "running", values, summary=dict(status="running"))
            result = report(root)
            self.assertEqual(result["case_count"], 0)
            self.assertEqual(result["groups"], [])
            self.assertEqual({row["case"] for row in result["excluded"]},
                             {"failed_energy", "hash_mismatch", "failed_run", "running"})
            self.assertTrue(all(row["reason"] for row in result["excluded"]))
            self.assertTrue(Path(result["manifest"]).is_file())

    def test_exposure_and_recorded_conditions_are_not_silently_mixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, duration in (("short_exposure", 200), ("long_exposure", 400)):
                save_case(root, name, [sample(0, 1, 500, 10), sample(duration, 1, 300, 50)])
            result = report(root)
            self.assertEqual(result["case_count"], 2)
            self.assertEqual(len(result["groups"]), 2)
            self.assertEqual({tuple(group["cases"]) for group in result["groups"]},
                             {("short_exposure",), ("long_exposure",)})

    def test_recorded_flight_conditions_get_separate_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, mach in (("mach8", 8.), ("mach9", 9.)):
                source = save_case(root, name, [sample(0, 1, 500, 10), sample(200, 1, 300, 50)])
                summary = dict(status="complete", energy_balance=dict(status="passed"),
                    settings=dict(duration=200., initial_t=300.),
                    surface_load_conditions=dict(flight=dict(mach=mach)),
                    result_sha256={source.name: file_sha256(source)})
                (source.parent / "thermal_summary.json").write_text(json.dumps(summary))
            result = report(root)
            self.assertEqual(result["case_count"], 2)
            self.assertEqual(len(result["groups"]), 2)
            self.assertEqual({tuple(group["cases"]) for group in result["groups"]}, {("mach8",), ("mach9",)})

    def test_nonfinite_and_duplicate_samples_are_not_reported_as_valid_peaks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_case(root, "nonfinite", [sample(0, 1, float("nan"), 10), sample(200, 1, 300, 50)])
            save_case(root, "duplicate", [sample(0, 1, 500, 10), sample(0, 1, 400, 20)])
            result = report(root)
            self.assertEqual(result["case_count"], 0)
            self.assertEqual({row["case"] for row in result["excluded"]}, {"nonfinite", "duplicate"})

    def test_cli_reports_archival_csv_without_native_clients_or_original_case_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs, output = root / "runs", root / "comparison"
            source = save_case(runs, "nose", [sample(0, 1, 500, 10), sample(200, 1, 300, 50)])
            before = source.read_bytes()
            environment = dict(os.environ, MPLCONFIGDIR=str(root / "matplotlib"), XDG_CACHE_HOME=str(root / "cache"))
            result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "study.py"),
                "report-heat-flux", "--runs-dir", str(runs), "--output-dir", str(output)],
                capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(list(output.rglob("peak_heat_flux_comparison.png")))
            self.assertTrue(list(output.rglob("peak_heat_flux_comparison.csv")))
            self.assertEqual(source.read_bytes(), before)

    def test_normal_comparison_entry_point_also_rebuilds_saved_flux_peaks(self):
        from compare_nose_trade import compare
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs, output = root / "runs", root / "comparison"
            source = save_case(runs, "nose", [sample(0, 1, 500, 10), sample(200, 1, 300, 50)])
            before = source.read_bytes()
            # Other renderers have independent provenance/figure tests. Keep
            # the real new loader and plot construction here so this catches
            # an orphaned feature that works only through its dedicated CLI.
            with patch("pipeline.gas_reporting.create_gas_reports", return_value=([], {})), \
                 patch("pipeline.mechanical_reporting.create_mechanical_reports", return_value=([], {})):
                compare(runs, output)
            manifest = output / "peak_heat_flux/peak_heat_flux_report.json"
            self.assertEqual(json.loads(manifest.read_text())["case_count"], 1)
            self.assertTrue((manifest.parent / "peak_heat_flux_comparison.png").is_file())
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__": unittest.main()
