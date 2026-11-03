"""Tests of the helpers of tools/verify_all.py."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "verify_all.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("verify_all_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(TOOL.exists(), "tools/ is not part of this install")
class VerifyAllHelpers(unittest.TestCase):
    def setUp(self):
        self.tool = _load_tool()

    def test_step_order_and_names(self):
        self.assertEqual(
            list(self.tool.STEPS),
            ["install", "environment", "ruff", "source", "mypy", "tests",
             "extras", "build", "twine", "clean_install", "equivalence",
             "examples"],
        )

    def test_version_matches_the_package(self):
        from clustelia import __version__

        self.assertEqual(self.tool.current_version(), __version__)

    def test_outcome_and_tail(self):
        self.assertEqual(self.tool.outcome(0, "ok")[0], "PASS")
        self.assertEqual(self.tool.outcome(2, "bad")[0], "FAIL")
        text = "\n".join(str(i) for i in range(100))
        self.assertEqual(self.tool.tail(text, 3), "97\n98\n99")

    def test_missing_tool_is_missing_not_pass(self):
        ctx = self.tool.Context(install=False, workdir=Path("."))
        self.assertEqual(self.tool.step_install(ctx)[0], "SKIPPED")
        original = self.tool.have
        self.tool.have = lambda module: False
        try:
            for step in (self.tool.step_ruff, self.tool.step_mypy,
                         self.tool.step_twine, self.tool.step_extras):
                with self.subTest(step=step.__name__):
                    self.assertEqual(step(ctx)[0], "MISSING")
        finally:
            self.tool.have = original

    def test_run_reports_failures_and_unknown_commands(self):
        code, output = self.tool.run(
            [self.tool.sys.executable, "-c", "raise SystemExit(3)"])
        self.assertEqual(code, 3)
        code, _ = self.tool.run(["definitely-not-a-command-xyz"])
        self.assertEqual(code, 127)

    def test_exit_codes_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.md"
            code = self.tool.main(["--only", "environment", "source",
                                   "--report", str(report)])
            self.assertEqual(code, 0)
            text = report.read_text(encoding="utf-8")
            self.assertIn("| source | PASS |", text)
            self.assertIn("## environment: PASS", text)
            code = self.tool.main(["--only", "extras", "--skip", "ruff",
                                   "--report", str(report)])
            self.assertIn(code, (0, 2))


if __name__ == "__main__":
    unittest.main()
