import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from cascabel.cli import main

TEST_CONLL = "Un\tO\nlook\tB-ENG\ntotal\tB-ENG\nblack\tI-ENG\n"
PRED_CONLL = "Un\tO\nlook\tB-ENG\ntotal\tO\nblack\tB-ENG\n"
TRAIN_CONLL = "Fracaso\tO\ndel\tO\nlook\tB-ENG\n"


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.test_path = self._write("test.conll", TEST_CONLL)
        self.pred_path = self._write("pred.conll", PRED_CONLL)
        self.train_path = self._write("train.conll", TRAIN_CONLL)

    def _write(self, name, content):
        path = Path(self.directory.name) / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    def _run(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_text_report(self):
        code, out, _ = self._run([self.test_path, self.pred_path])
        self.assertEqual(code, 0)
        self.assertIn("cascabel report", out)
        self.assertIn("overlap_missed", out)

    def test_json_report(self):
        code, out, _ = self._run(
            [self.test_path, self.pred_path, "--training", self.train_path,
             "--format", "json"]
        )
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["error_counts"]["overlap_missed"], 1)
        self.assertEqual(sum(report["error_counts"].values()), 1)
        self.assertEqual(report["error_counts"]["fused"], 0)  # zeros listed
        self.assertIn("unseen", report["dimensions"])

    def test_html_to_file(self):
        output = str(Path(self.directory.name) / "report.html")
        code, out, err = self._run(
            [self.test_path, self.pred_path, "--format", "html", "-o", output]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out, "")
        self.assertIn("report written", err)
        content = Path(output).read_text(encoding="utf-8")
        self.assertIn("<!doctype html>", content)

    def test_profile_only(self):
        code, out, _ = self._run([self.test_path])
        self.assertEqual(code, 0)
        self.assertIn("Dataset:", out)
        self.assertNotIn("Errors", out)

    def test_missing_file_error(self):
        code, _, err = self._run(["does-not-exist.conll"])
        self.assertEqual(code, 2)
        self.assertIn("error", err)

    def test_misaligned_prediction_error(self):
        bad_pred = self._write("bad.conll", "Un\tO\n")
        code, _, err = self._run([self.test_path, bad_pred])
        self.assertEqual(code, 2)
        self.assertIn("token", err)


if __name__ == "__main__":
    unittest.main()
