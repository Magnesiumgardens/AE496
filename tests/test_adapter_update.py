from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.update_adapters import FILES, update


class AdapterUpdateTests(unittest.TestCase):
    revision = "a" * 40

    def project(self, root):
        (root / "study.py").write_text("# entrypoint\n")
        for relative in FILES:
            path = root / relative
            path.parent.mkdir(exist_ok=True)
            path.write_text("# previous local adapter\n")
        for relative in ("config.yaml", "runs/blunt_1/04_final.dat.h5"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"preserve this exact content")

    def test_complete_update_keeps_backups_and_preserves_case_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.project(root)
            backup = update(root, self.revision, fetch=lambda *args: b"# corrected adapter\n")
            for relative in FILES:
                self.assertEqual((root / relative).read_bytes(), b"# corrected adapter\n")
                self.assertEqual((backup / relative).read_bytes(), b"# previous local adapter\n")
            for relative in ("config.yaml", "runs/blunt_1/04_final.dat.h5"):
                self.assertEqual((root / relative).read_bytes(), b"preserve this exact content")
            self.assertIn(self.revision, (backup / "update.json").read_text())
            self.assertEqual(list(root.glob("adapter-update-*")), [])

    def test_late_download_or_syntax_failure_changes_no_source(self):
        for failure in (OSError("download interrupted"), b"try:\n"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.project(root)
                def fetch(revision, relative):
                    if relative == FILES[-1]:
                        if isinstance(failure, Exception):raise failure
                        return failure
                    return b"# corrected adapter\n"
                with self.assertRaises((OSError, SyntaxError)):
                    update(root, self.revision, fetch=fetch)
                for relative in FILES:
                    self.assertEqual((root / relative).read_bytes(), b"# previous local adapter\n")
                self.assertFalse((root / "logs").exists())

    def test_partial_install_is_rolled_back(self):
        import os
        replace = os.replace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.project(root)
            def fail_third(source, target):
                if Path(target) == root / FILES[2]:raise OSError("file locked")
                replace(source, target)
            with patch("scripts.update_adapters.os.replace", side_effect=fail_third):
                with self.assertRaisesRegex(OSError, "file locked"):
                    update(root, self.revision, fetch=lambda *args: b"# corrected adapter\n")
            for relative in FILES:
                self.assertEqual((root / relative).read_bytes(), b"# previous local adapter\n")


if __name__ == "__main__":
    unittest.main()
