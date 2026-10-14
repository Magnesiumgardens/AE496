"""Exercise the stage watchdog against real subprocess and native file writes."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline.study import BatchError, run_stage


class StageProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.native = self.root / "nose_thermal.out"
        self.log = self.root / "thermal.log"

    def stage(self, script, *, hard=4.0, inactivity=0.5, files=None):
        with redirect_stdout(io.StringIO()):
            return run_stage([sys.executable, "-u", "-c", script, str(self.native)],
                             self.log, hard_timeout_s=hard, inactivity_timeout_s=inactivity,
                             activity_files=files if files is not None else [self.native])

    def test_silent_native_file_growth_keeps_stage_active(self):
        self.stage("""
from pathlib import Path
import sys, time
with Path(sys.argv[1]).open('w') as stream:
    for step in range(14):
        stream.write(f'result set {step}\\n')
        stream.flush()
        time.sleep(0.08)
""")
        self.assertIn("result set 13", self.native.read_text())
        self.assertIn("Native activity", self.log.read_text())

    def test_existing_stationary_file_does_not_count_as_progress(self):
        self.native.write_text("previous attempt completed\n")
        with self.assertRaisesRegex(BatchError, "Stage inactivity timeout"):
            self.stage("import time; time.sleep(4)")
        self.assertNotIn("Native activity", self.log.read_text())

    def test_new_file_that_stops_changing_still_times_out(self):
        with self.assertRaisesRegex(BatchError, "last native activity"):
            self.stage("from pathlib import Path; import sys, time; "
                       "Path(sys.argv[1]).write_text('startup only'); time.sleep(4)")
        self.assertIn("Stage inactivity timeout", self.log.read_text())

    def test_growth_does_not_extend_hard_deadline_and_owned_process_is_stopped(self):
        launched = []
        real_popen = subprocess.Popen

        def track_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            launched.append(process)
            return process

        with patch("pipeline.study.subprocess.Popen", side_effect=track_process):
            with self.assertRaisesRegex(BatchError, "Stage hard timeout"):
                self.stage("""
from pathlib import Path
import sys, time
with Path(sys.argv[1]).open('w') as stream:
    while True:
        stream.write('computing\\n')
        stream.flush()
        time.sleep(0.06)
""", hard=0.8)
        self.assertEqual(len(launched), 1)
        self.assertIsNotNone(launched[0].poll())
        self.assertTrue(launched[0].stdout.closed)
        self.assertIn("last native activity", self.log.read_text())

    def test_regular_stdout_remains_activity_without_watched_files(self):
        self.stage("""
import time
for step in range(12):
    print(f'APDL step {step}', flush=True)
    time.sleep(0.08)
""", files=[])
        self.assertIn("APDL step 11", self.log.read_text())

    def test_unwatched_file_growth_cannot_keep_stage_alive(self):
        with self.assertRaisesRegex(BatchError, "Stage inactivity timeout"):
            self.stage("""
from pathlib import Path
import sys, time
with Path(sys.argv[1]).open('w') as stream:
    while True:
        stream.write('unrelated output\\n')
        stream.flush()
        time.sleep(0.06)
""", files=[])

    @unittest.skipIf(os.name == "nt", "POSIX process-group cleanup test")
    def test_timeout_stops_owned_native_child_process(self):
        stopped = self.root / "child_stopped.txt"
        child = """
from pathlib import Path
import signal, sys, time
def stop(signum, frame):
    Path(sys.argv[1]).write_text('stopped')
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
while True:
    time.sleep(0.05)
"""
        parent = f"""
import subprocess, sys, time
child = subprocess.Popen([sys.executable, '-u', '-c', {child!r}, {str(stopped)!r}])
time.sleep(4)
"""
        with self.assertRaisesRegex(BatchError, "Stage inactivity timeout"):
            self.stage(parent, inactivity=0.7)
        self.assertEqual(stopped.read_text(), "stopped")


if __name__ == "__main__":
    unittest.main()
