"""Convenience entry point keeps the intended exposure and corrected names."""
import unittest
from unittest.mock import patch

from run_internal_170 import main


class Internal170EntryTests(unittest.TestCase):
    def test_default_runs_170_seconds_and_report_mode_uses_the_same_physical_settings(self):
        with patch("run_internal_170.compare_internal",return_value={"status":"complete"}) as run:
            self.assertEqual(main([]),0)
            options = run.call_args.kwargs
            self.assertEqual(options["duration_s"],170.)
            self.assertEqual(options["processors"],1)
            self.assertFalse(options["report_only"])
            self.assertEqual(options["swap_labels"],["mk3_body_internals_base_nose1","mk3_body_internals_blunt_nose"])
            self.assertEqual(main(["--report-only"]),0)
            self.assertEqual(run.call_args.kwargs,{**options,"report_only":True})


if __name__ == "__main__":unittest.main()
