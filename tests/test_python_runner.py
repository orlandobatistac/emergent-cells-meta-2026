"""Python-only experiment runner smoke tests; no browser, HTTP or Meta API."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from emergent_cells.cli import EXPERIMENTS, main


class PythonRunnerTests(unittest.TestCase):
    def setUp(self):
        self.original = sys.argv

    def tearDown(self):
        sys.argv = self.original

    def test_list_experiments(self):
        sys.argv = ["emergent-cells", "list"]
        with io.StringIO() as out, contextlib.redirect_stdout(out):
            main()
            value = out.getvalue()
        for experiment in ("dynamic-memory", "learned-curiosity",
                           "active-probes", "messages", "simulate"):
            self.assertIn(experiment, value)
        self.assertGreaterEqual(len(EXPERIMENTS), 10)

    def test_simulation_reproducible_json(self):
        with tempfile.TemporaryDirectory() as workspace:
            out = Path(workspace) / "run.json"
            sys.argv = ["emergent-cells", "simulate", "--seed", "11",
                        "--steps", "8", "--cells", "12", "--output", str(out)]
            with contextlib.redirect_stdout(io.StringIO()):
                main()
            before = out.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(before, out.read_bytes())
            self.assertTrue(json.loads(before))

    def test_unknown_command_rejected(self):
        sys.argv = ["emergent-cells", "web"]
        with self.assertRaises(SystemExit):
            main()

    def test_dispatched_experiment_help(self):
        sys.argv = ["emergent-cells", "dynamic-memory", "--help"]
        with self.assertRaises(SystemExit) as end, contextlib.redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(end.exception.code, 0)
        self.assertEqual(sys.argv[1], "dynamic-memory")


if __name__ == "__main__":
    unittest.main()
