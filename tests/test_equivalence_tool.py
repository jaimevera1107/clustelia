"""Tests of the cross-environment tolerance mode of tools/equivalence.py."""

import contextlib
import importlib.util
import io
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TOOL = Path(__file__).resolve().parents[1] / "tools" / "equivalence.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("equivalence_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(shift=0.0, status="recommended", labels=(0, 1, 1)):
    table = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "silhouette": [0.5 + shift, np.nan],
        "structure_test": [{"observed": 0.5 + shift, "grid": ["a"]}, None],
    })
    return {
        "status": status, "recommendation": "a", "dropped": [],
        "table": table, "labels": {"a": np.array(labels), "b": "Error"},
    }


@unittest.skipUnless(TOOL.exists(), "tools/ is not part of this install")
class ToleranceMode(unittest.TestCase):
    def setUp(self):
        self.tool = _load_tool()

    def _compare(self, x, y, rtol):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for name, snap in (("a.pkl", x), ("b.pkl", y)):
                path = Path(folder) / name
                path.write_bytes(pickle.dumps({("s", "p"): snap}))
                paths.append(str(path))
            with contextlib.redirect_stdout(io.StringIO()):
                return self.tool.compare(*paths, rtol=rtol)

    def test_last_bit_differences_pass_only_with_a_tolerance(self):
        x, y = _snapshot(), _snapshot(shift=1e-14)
        self.assertEqual(self._compare(x, y, None), 1)
        self.assertEqual(self._compare(x, y, 1e-9), 0)

    def test_real_differences_still_fail_with_a_tolerance(self):
        base = _snapshot()
        for changed in (_snapshot(shift=1e-3),
                        _snapshot(status="ambiguous"),
                        _snapshot(labels=(0, 1, 0))):
            with self.subTest():
                self.assertEqual(self._compare(base, changed, 1e-9), 1)

    def test_nested_floats_use_the_tolerance_but_ids_do_not(self):
        close = self.tool.values_close
        self.assertTrue(close({"a": [1.0, np.nan]},
                              {"a": [1.0 + 1e-13, np.nan]}, 1e-9))
        self.assertFalse(close({"a": [1.0]}, {"a": [1.1]}, 1e-9))
        self.assertFalse(close("abc", "abd", 1e-3))
        self.assertFalse(close(1, 2, 1.0))
        self.assertTrue(close(None, None, 1e-9))

    def test_table_shape_and_columns_must_match(self):
        x = pd.DataFrame({"u": [1.0, 2.0]})
        self.assertFalse(self.tool.tables_close(
            x, pd.DataFrame({"v": [1.0, 2.0]}), 1e-9))
        self.assertFalse(self.tool.tables_close(
            x, pd.DataFrame({"u": [1.0]}), 1e-9))

    def test_command_line_rejects_a_nonpositive_tolerance(self):
        with self.assertRaises(SystemExit), \
                contextlib.redirect_stderr(io.StringIO()):
            self.tool.main(["compare", "a", "b", "--rtol", "0"])


if __name__ == "__main__":
    unittest.main()
