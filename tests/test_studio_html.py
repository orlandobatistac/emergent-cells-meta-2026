"""Catch broken Studio markup references and JavaScript syntax before demos."""
import re
import shutil
import subprocess
import unittest
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / (
    "src/emergent_cells/experiments/studio.html"
)


class StudioBrowserAssetTests(unittest.TestCase):
    def test_controls_have_unique_dom_ids(self):
        html = PAGE.read_text(encoding="utf-8")
        ids = re.findall(r'\bid="([^"]+)"', html)
        self.assertEqual(len(ids), len(set(ids)))
        for target in ("grid", "reset", "compare", "trials", "provider",
                       "download-evidence", "evidence-result", "meta-state"):
            self.assertIn(target, ids)

    @unittest.skipUnless(shutil.which("node"), "Node.js unavailable")
    def test_inline_javascript_parses(self):
        html = PAGE.read_text(encoding="utf-8")
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        result = subprocess.run(["node", "--check", "-"], input=script,
                                text=True, encoding="utf-8", capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
