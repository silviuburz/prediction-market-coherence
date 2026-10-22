"""Regression checks for the text-free numeric auditor."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "tools" / "check_numeric_claims.py"


class NumericClaimTests(unittest.TestCase):
    def make_package(self, path):
        shutil.copytree(ROOT / "figures" / "data", path / "figures" / "data")
        return path

    def audit(self, package, *options):
        return subprocess.run(
            [sys.executable, str(AUDITOR), "--package", str(package), *options],
            capture_output=True, text=True, encoding="utf-8")

    def test_all_calculations_work_without_paper_text(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.make_package(Path(directory))
            result = self.audit(package, "--emit-map", str(package / "claims.json"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            record = json.loads((package / "claims.json").read_text())
            self.assertEqual(record["check_count"], 354)
            self.assertTrue(all(check["passed"] for check in record["checks"]))
            self.assertNotIn("manuscript", record)
            self.assertNotIn("supplement", record)
            for artifact in record["artifacts"]:
                path = package / artifact["path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 artifact["sha256"])

    def test_changed_primary_estimate_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.make_package(Path(directory))
            path = package / "figures" / "data" / "corrected_specification.json"
            data = json.loads(path.read_text())
            next(row for row in data["runs"]
                 if row["cohort"] == "holdout")["D_cents"] = 0.5
            path.write_text(json.dumps(data))
            result = self.audit(package)
            self.assertEqual(result.returncode, 1)
            self.assertIn("corrected holdout D", result.stdout)

    def test_changed_event_count_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.make_package(Path(directory))
            path = package / "figures" / "data" / "event_funnel.json"
            data = json.loads(path.read_text())
            data["source_population"]["regulation_goals"] = 299
            path.write_text(json.dumps(data))
            result = self.audit(package)
            self.assertEqual(result.returncode, 1)
            self.assertIn("funnel source goals", result.stdout)

    def test_missing_input_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self.make_package(Path(directory))
            (package / "figures" / "data" / "corrected_specification.json").unlink()
            result = self.audit(package)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("corrected_specification.json", result.stderr)
            self.assertNotIn("all numeric claims", result.stdout)


if __name__ == "__main__":
    unittest.main()
