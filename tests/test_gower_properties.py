"""Property-based tests of the weighted Gower distance (hypothesis)."""

import unittest

import numpy as np
import pandas as pd

try:
    from hypothesis import HealthCheck, assume, given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - dev extra
    raise unittest.SkipTest("hypothesis is not installed") from None

from clustelia.geometry import GowerMetric

LEVELS = ("low", "mid", "high")
SETTINGS = settings(max_examples=60, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


@st.composite
def mixed_frames(draw: st.DrawFn) -> tuple[pd.DataFrame, dict[str, float]]:
    """Frames with numeric, nominal, ordinal and binary columns."""
    rows = draw(st.integers(3, 12))
    floats = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
    frame = pd.DataFrame({
        "x": draw(st.lists(floats, min_size=rows, max_size=rows)),
        "c": draw(st.lists(st.sampled_from("abc"), min_size=rows,
                           max_size=rows)),
        "o": draw(st.lists(st.sampled_from(LEVELS), min_size=rows,
                           max_size=rows)),
        "b": draw(st.lists(st.integers(0, 1), min_size=rows,
                           max_size=rows)),
    })
    weights = {c: draw(st.floats(0.1, 10.0)) for c in frame.columns}
    return frame, weights


def _metric(frame: pd.DataFrame, weights: dict[str, float] | None = None,
            ) -> tuple[GowerMetric, pd.DataFrame]:
    metric = GowerMetric(["x"], ["c"], {"o": LEVELS}, ["b"], weights)
    metric.fit(frame)
    return metric, metric.transform(frame)


def _naive(frame: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Pair-by-pair transcription of Gower (1971) with asymmetric binary."""
    span = frame["x"].max() - frame["x"].min()
    positions = {level: i / (len(LEVELS) - 1) for i, level in
                 enumerate(LEVELS)}
    n = len(frame)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            num = den = 0.0
            r, s = frame.iloc[i], frame.iloc[j]
            terms = [
                ("x", min(abs(r.x - s.x) / span, 1.0), True),
                ("c", float(r.c != s.c), True),
                ("o", abs(positions[r.o] - positions[s.o]), True),
                ("b", float(r.b != s.b), bool(r.b or s.b)),
            ]
            for column, value, comparable in terms:
                if comparable:
                    num += weights[column] * value
                    den += weights[column]
            out[i, j] = num / den if den else 0.0
    return out


class GowerProperties(unittest.TestCase):
    @SETTINGS
    @given(mixed_frames())
    def test_matches_pairwise_definition(self, case):
        frame, weights = case
        assume(frame["x"].nunique() > 1)
        metric, coded = _metric(frame, weights)
        np.testing.assert_allclose(metric.distances(coded),
                                   _naive(frame, weights), atol=1e-12)

    @SETTINGS
    @given(mixed_frames())
    def test_symmetric_bounded_zero_diagonal(self, case):
        frame, weights = case
        assume(frame["x"].nunique() > 1)
        metric, coded = _metric(frame, weights)
        D = metric.distances(coded)
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        self.assertTrue(((D >= 0) & (D <= 1 + 1e-12)).all())
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    @SETTINGS
    @given(mixed_frames(), st.floats(0.1, 50.0))
    def test_invariant_to_uniform_weight_scaling(self, case, scale):
        frame, _ = case
        assume(frame["x"].nunique() > 1)
        metric_1, coded = _metric(frame)
        metric_s, _ = _metric(frame, {c: scale for c in frame.columns})
        np.testing.assert_allclose(metric_1.distances(coded),
                                   metric_s.distances(coded), atol=1e-12)

    @SETTINGS
    @given(mixed_frames())
    def test_binary_joint_absence_is_ignored(self, case):
        frame, weights = case
        assume(frame["x"].nunique() > 1)
        absent = frame.assign(b=0)
        metric, coded = _metric(absent, weights)
        reduced = GowerMetric(["x"], ["c"], {"o": LEVELS}, [], weights)
        reduced.fit(absent)
        np.testing.assert_allclose(
            metric.distances(coded),
            reduced.distances(reduced.transform(absent)), atol=1e-12)

    @SETTINGS
    @given(mixed_frames())
    def test_cross_distances_match_square_block(self, case):
        frame, weights = case
        assume(frame["x"].nunique() > 1 and len(frame) > 3)
        metric, coded = _metric(frame, weights)
        full = metric.distances(coded)
        block = metric.distances(coded.iloc[:2], coded.iloc[2:])
        np.testing.assert_allclose(block, full[:2, 2:], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
