"""Updater installs internal-study inputs once and retains physical choices."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.update_project import update_project


class InternalScreeningUpdateTests(unittest.TestCase):
    def test_first_install_distributes_table_and_config_then_preserves_exact_user_input(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, target = base / "source", base / "user"
            source.mkdir(); target.mkdir()
            (source / "materials").mkdir()
            for filename in ("config.yaml", "thermal_analysis.yaml", "batch.yaml", "internal_screening.yaml",
                             "materials/reference_carbon_carbon_2d.yaml"):
                shutil.copyfile(repository / filename, source / filename)
            (source / "study.py").write_text("# Distribution entry point\n", encoding="utf-8")
            (target / "study.py").write_text("# Existing local entry point\n", encoding="utf-8")

            def git(*args):
                return subprocess.check_output(["git", "-C", str(source), *args], stderr=subprocess.STDOUT)

            def commit(message):
                git("add", ".")
                git("-c", "user.name=Update Test", "-c", "user.email=update-test@example.invalid",
                    "commit", "--quiet", "-m", message)
                return git("rev-parse", "HEAD").decode().strip()

            git("init", "--quiet")
            first_revision = commit("Initial distribution")
            with redirect_stdout(io.StringIO()):
                update_project(target, source, first_revision)
            for filename in ("internal_screening.yaml", "materials/reference_carbon_carbon_2d.yaml"):
                self.assertEqual((target / filename).read_bytes(), (source / filename).read_bytes())

            # Comments, ordering and explicit physical settings all survive.
            customized = (b"# Keep the confirmed measurement convention and material axes.\r\n"
                          b"depth_mode: axial_tip\r\nmaterial_axes_2d: [x, z]\r\n"
                          b"duration_s: 400\r\ndepth_m: 0.010\r\n")
            (target / "internal_screening.yaml").write_bytes(customized)
            (source / "internal_screening.yaml").write_text("depth_mode: surface_normal\nduration_s: 200\n", encoding="utf-8")
            table = source / "materials/reference_carbon_carbon_2d.yaml"
            table.write_bytes(table.read_bytes() + b"\n# Revised distribution table documentation.\n")
            second_revision = commit("Revised defaults and table documentation")
            with redirect_stdout(io.StringIO()):
                backup = update_project(target, source, second_revision)
            self.assertEqual((target / "internal_screening.yaml").read_bytes(), customized)
            self.assertEqual((target / "materials/reference_carbon_carbon_2d.yaml").read_bytes(), table.read_bytes())
            manifest = json.loads((backup / "update.json").read_text())
            self.assertNotIn("internal_screening.yaml", manifest["files"])
            self.assertIn("materials/reference_carbon_carbon_2d.yaml", manifest["files"])


if __name__ == "__main__":unittest.main()
