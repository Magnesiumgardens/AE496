"""Only energy-checked radiation solves are eligible for current comparisons."""
from contextlib import contextmanager
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline.mechanical_reporting import mechanical_complete
from pipeline.thermal import AEROTHERMAL_MODEL_VERSION
from pipeline.utils import atomic_json, file_sha256
from test_aerothermal_reports import saved_aerothermal


@contextmanager
def current_case():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        work = root / "nose"
        summary = saved_aerothermal(work)
        thermal = root / "thermal.yaml"
        thermal.write_text("loading: {mode: aerodynamic_surface_load}\n")
        boundary = root / "surface_load_nodes.csv"
        boundary.write_text("verified calibration boundary")
        package = root / "surface_load_summary.json"
        package.write_text("verified calibration metadata")
        metadata = dict(source_sha256={"re_case": "a", "re_data": "b", "cold_case": "c", "cold_data": "d"},
                        conditions={"thermal": {"mode": "radiative_equilibrium"}})
        summary.update(thermal_config_path=str(thermal), thermal_config_sha256=file_sha256(thermal),
                       boundary_csv_sha256=file_sha256(boundary), source_sha256=metadata["source_sha256"],
                       surface_package_sha256=file_sha256(package), surface_load_conditions=metadata["conditions"])
        path = work / "thermal_analysis/thermal_summary.json"
        atomic_json(path, summary)
        cfg = {"project": {"work_dir": str(work)}, "thermal": {"mode": "radiative_equilibrium"},
               "workflow": {"strategy": "single_solve"}}
        with patch("pipeline.aerothermal_calibration.verify_surface_package", return_value=(boundary, metadata)), \
             patch("pipeline.aerothermal_calibration.calibration_paths", return_value={"summary": package}):
            yield work, cfg, summary, path


class RadiationProvenanceTests(unittest.TestCase):
    def test_current_surf151_energy_checked_results_are_eligible(self):
        self.assertEqual(AEROTHERMAL_MODEL_VERSION, "7.0-surf151-aerodynamic-surface-load")
        with current_case() as (work, cfg, summary, path):
            self.assertTrue(mechanical_complete(work, cfg))
            # Old results must not regain eligibility merely because their
            # source files and result hashes still match.
            summary["model_version"] = "6.0-aerodynamic-surface-load"
            atomic_json(path, summary)
            self.assertFalse(mechanical_complete(work, cfg))

    def test_missing_failed_or_inconsistent_energy_checks_are_rejected(self):
        with current_case() as (work, cfg, summary, path):
            for status in (None, "failed", "not_verified"):
                changed = copy.deepcopy(summary)
                if status is None:
                    changed.pop("energy_balance")
                else:
                    changed["energy_balance"]["status"] = status
                atomic_json(path, changed)
                self.assertFalse(mechanical_complete(work, cfg), status)
            atomic_json(path, summary)
            balance_path = work / "thermal_analysis/energy_balance.json"
            atomic_json(balance_path, {**summary["energy_balance"], "relative_tolerance": .5})
            # Even an otherwise matching hash must not allow contradictory
            # summary and standalone verification records.
            summary["result_sha256"][balance_path.name] = file_sha256(balance_path)
            atomic_json(path, summary)
            self.assertFalse(mechanical_complete(work, cfg))

    def test_radiation_configuration_and_verified_energy_artifacts_are_required(self):
        with current_case() as (work, cfg, summary, path):
            for radiation in ({}, {"formulation": "SFL_RAD", "status": "configured"},
                              {"formulation": "SURF151", "status": "not_verified"}):
                changed = dict(summary, radiation_boundary=radiation)
                atomic_json(path, changed)
                self.assertFalse(mechanical_complete(work, cfg), radiation)
            for name in ("solid_energy_history.csv", "energy_balance.json"):
                changed = copy.deepcopy(summary)
                del changed["result_sha256"][name]
                atomic_json(path, changed)
                self.assertFalse(mechanical_complete(work, cfg), name)
            atomic_json(path, summary)
            energy_path = work / "thermal_analysis/solid_energy_history.csv"
            energy_path.write_text("altered stored-energy history")
            self.assertFalse(mechanical_complete(work, cfg))


if __name__ == "__main__":unittest.main()
