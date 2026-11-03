"""Gower pairwise deletion of missing values (GeometrySpec.missing)."""

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from clustelia import AutoCluster, GeometrySpec, Schema
from clustelia.geometry import GowerMetric

LEVELS = ("low", "mid", "high")
# Explicit handling when random missingness leaves two rows incomparable.
PAIRWISE = GeometrySpec(kind="gower", scaling="none", missing="pairwise",
                        incomparable="max")


def _frame(seed: int = 0, rows: int = 120) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.repeat([0, 1, 2], rows // 3)
    frame = pd.DataFrame({
        "x": rng.normal(groups * 5.0, 0.6),
        "colour": np.array(["red", "green", "blue"], dtype=object)[groups],
        "level": np.array(LEVELS, dtype=object)[groups],
        "flag": np.array([0, 1, 1])[groups],
    })
    return frame, groups


def _with_missing(frame: pd.DataFrame, fraction: float,
                  seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = frame.astype(object).copy()
    out["x"] = frame["x"].astype(float)
    mask = rng.random(out.shape) < fraction
    mask[:, 0] &= ~mask[:, 1:].all(axis=1)  # never blank a whole row
    for position, column in enumerate(out.columns):
        out.loc[mask[:, position], column] = np.nan
    out["x"] = out["x"].astype(float)
    out["flag"] = out["flag"].astype(float)
    return out


SCHEMA = Schema(numeric=["x"], categorical=["colour"],
                ordinal={"level": LEVELS}, binary=["flag"])


def _naive(frame: pd.DataFrame) -> np.ndarray:
    span = frame["x"].max() - frame["x"].min()
    position = {level: i / 2 for i, level in enumerate(LEVELS)}
    n = len(frame)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r, s = frame.iloc[i], frame.iloc[j]
            num = den = 0.0
            if pd.notna(r.x) and pd.notna(s.x):
                num += min(abs(r.x - s.x) / span, 1.0)
                den += 1
            if pd.notna(r.colour) and pd.notna(s.colour):
                num += float(r.colour != s.colour)
                den += 1
            if pd.notna(r.level) and pd.notna(s.level):
                num += abs(position[r.level] - position[s.level])
                den += 1
            if pd.notna(r.flag) and pd.notna(s.flag) and (r.flag or s.flag):
                num += float(r.flag != s.flag)
                den += 1
            out[i, j] = num / den if den else 0.0
    return out


def _metric(frame: pd.DataFrame) -> tuple[GowerMetric, pd.DataFrame]:
    metric = GowerMetric(["x"], ["colour"], {"level": LEVELS}, ["flag"],
                         incomparable="zero")
    metric.fit(frame)
    return metric, metric.transform(frame)


class PairwiseDistance(unittest.TestCase):
    def test_matches_definition_with_missing_values(self):
        frame = _with_missing(_frame(rows=30)[0], 0.25)
        metric, coded = _metric(frame)
        np.testing.assert_allclose(metric.distances(coded), _naive(frame),
                                   atol=1e-12)

    def test_pairwise_path_equals_complete_path_without_missing(self):
        frame = _frame(rows=30)[0]
        metric, coded = _metric(frame)
        np.testing.assert_allclose(metric._pairwise_distances(coded, coded),
                                   metric.distances(coded), atol=1e-15)

    def test_distances_stay_symmetric_and_bounded(self):
        frame = _with_missing(_frame(rows=30)[0], 0.4)
        metric, coded = _metric(frame)
        D = metric.distances(coded)
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        self.assertTrue(((D >= 0) & (D <= 1)).all())


class PairwisePolicy(unittest.TestCase):
    def test_default_policy_still_rejects_missing(self):
        frame = _with_missing(_frame()[0], 0.1)
        with self.assertRaisesRegex(ValueError, "missing='pairwise'"):
            AutoCluster(schema=SCHEMA, k_range=(2, 3)).fit(frame)

    def test_pairwise_requires_gower(self):
        with self.assertRaisesRegex(ValueError, "Gower geometry only"):
            GeometrySpec(kind="euclidean", missing="pairwise")
        with self.assertRaises(ValueError):
            GeometrySpec(kind="gower", scaling="none", missing="drop")

    def test_rows_without_observed_features_are_rejected(self):
        frame = _with_missing(_frame()[0], 0.1)
        frame.loc[0, :] = np.nan
        with self.assertRaisesRegex(ValueError, "at least one observed"):
            AutoCluster(schema=SCHEMA, geometry=PAIRWISE,
                        k_range=(2, 3)).fit(frame)

    def test_fit_and_predict_recover_structure_with_missing(self):
        complete, groups = _frame()
        frame = _with_missing(complete, 0.15)
        model = AutoCluster(schema=SCHEMA, geometry=PAIRWISE,
                            k_range=(2, 3, 4), random_state=0).fit(frame)
        result = model.result_
        self.assertEqual(result.selection_status, "recommended")
        labels = result.select(result.recommendation.candidate_id).labels_
        self.assertGreater(adjusted_rand_score(groups, labels), 0.9)
        new = frame.iloc[:5].copy()
        new.loc[new.index[0], "colour"] = np.nan
        self.assertEqual(len(model.predict(new)), 5)
        self.assertEqual(result.metadata["geometry"]["missing"], "pairwise")
        profiles = result.select(result.recommendation.candidate_id
                                 ).describe()
        self.assertTrue(any("missing" in p for p in profiles))
        for profile in profiles:
            self.assertTrue(0 <= profile["binary"]["flag"] <= 1)

    def test_pairwise_on_complete_data_matches_error_policy(self):
        frame, _ = _frame()
        geometry_error = GeometrySpec(kind="gower", scaling="none")
        a = AutoCluster(schema=SCHEMA, geometry=geometry_error,
                        k_range=(2, 3), random_state=0).fit(frame).result_
        b = AutoCluster(schema=SCHEMA, geometry=PAIRWISE,
                        k_range=(2, 3), random_state=0).fit(frame).result_
        self.assertEqual(a.selection_status, b.selection_status)
        columns = ["candidate_id", "silhouette", "confirmed"]
        pd.testing.assert_frame_equal(a.candidates_table()[columns],
                                      b.candidates_table()[columns])


if __name__ == "__main__":
    unittest.main()
