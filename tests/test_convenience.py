"""Regressions for the convenience API and readable outputs."""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from clustelia import AutoCluster, Schema, cluster, detect_identifiers
from clustelia.config import SearchBudget, SelectionPolicy
from clustelia.geometry import GowerMetric
from clustelia.results import ExperimentResult


class ConvenienceTests(unittest.TestCase):
    def test_identifier_inference_is_conservative(self):
        f = pd.DataFrame({
            "customer_id": ["a", "b", "c", "d"],
            "score": [10, 11, 12, 13],
            "code": ["x", "y", "z", "w"],
            "other_id": ["a", "a", "c", "d"],
        })
        self.assertEqual(detect_identifiers(f), ("customer_id",))
        missing = f.assign(customer_id=[None, "b", "c", "d"])
        self.assertEqual(detect_identifiers(missing), ())
        with self.assertRaises(TypeError):
            detect_identifiers(f.to_numpy())

    def test_validation_and_effort(self):
        X, _ = make_blobs(n_samples=65, n_features=2, centers=3,
                          cluster_std=.4, random_state=2)
        f = pd.DataFrame(X, columns=["a", "b"])
        f.insert(0, "entity_id", [f"row-{i}" for i in range(len(f))])
        estimator = AutoCluster(schema=Schema(numeric=("a", "b"),
                                              identifiers=("entity_id",)),
                                selection=SelectionPolicy.p0(),
                                k_range=(2, 3), families=("kmeans",),
                                budget=SearchBudget(max_fits=100))
        plan = estimator.plan(f)
        self.assertEqual(plan["schema"]["identifiers"], ("entity_id",))
        res = cluster(f, effort="quick", families=("kmeans",),
                      k_range=(2, 3), random_state=42)
        self.assertIsInstance(res, ExperimentResult)
        self.assertEqual(res.summary()["excluded_identifiers"], ["entity_id"])
        self.assertEqual(res.metadata["effort"], "quick")
        self.assertIn("Selection:", res.report())
        self.assertEqual(res.ranking_table(top=1).shape[0], 1)
        result = cluster(f, effort="balanced", families=("kmeans",),
                         k_range=(2, 3), selection=SelectionPolicy.p0())
        self.assertEqual(result.metadata["effort"], "balanced")
        with self.assertRaisesRegex(ValueError, "effort"):
            cluster(f, effort="light")
        with self.assertRaisesRegex(ValueError,
                                    "either schema or identifiers"):
            cluster(f, schema=Schema(numeric=("a", "b"),
                                     identifiers=("entity_id",)),
                    identifiers=("entity_id",))

    def test_report_empty_result_and_profile(self):
        empty = ExperimentResult("incomplete", (), {}, None,
                                 {"objective": "compact"})
        self.assertEqual(empty.summary()["candidates_evaluated"], 0)
        self.assertIsNone(empty.recommended_solution())
        self.assertIn("No candidates evaluated", empty.report())
        X, _ = make_blobs(n_samples=55, centers=2, n_features=2,
                          random_state=4, cluster_std=.35)
        f = pd.DataFrame(X, columns=["x", "y"])
        e = AutoCluster(selection=SelectionPolicy.p0(),
                        families=("kmeans",), k_range=(2,),
                        random_state=0).fit(f)
        result = e.result_
        solution = result.select(result.candidate_ids[0])
        self.assertEqual(e.summary(), result.summary())
        self.assertEqual(e.report(), result.report())
        self.assertEqual(len(e.ranking_table()),
                         len(result.ranking_table()))
        table = solution.profile_table()
        self.assertEqual(set(table["feature"]), {"x", "y"})
        self.assertEqual(solution.sizes_table()["size"].sum(), len(f))
        self.assertTrue(all(table["median"].notna()))
        self.assertEqual(len(solution.labels_), len(f))
        if result.recommendation:
            self.assertEqual(result.recommended_solution().candidate_id,
                             result.recommendation.candidate_id)
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            return
        self.assertIsNotNone(solution.plot_sizes())
        self.assertIsNotNone(solution.plot_profile("x"))
        self.assertIsNotNone(result.plot_candidates())


class VectorizedGowerTests(unittest.TestCase):
    @staticmethod
    def scalar(a, b, ranges, weights):
        values = []
        for i in range(len(a)):
            row = []
            for j in range(len(b)):
                numerator = denominator = 0.0
                for c, kind in (("v", "numeric"), ("cat", "nominal"),
                                ("ord", "ordinal"), ("bin", "binary")):
                    x, y = a.iloc[i][c], b.iloc[j][c]
                    joint_absence = kind == "binary" and x == y == 0
                    if pd.isna(x) or pd.isna(y) or joint_absence:
                        continue
                    if kind == "numeric":
                        value = min(abs(x - y) / ranges[c], 1.0)
                    elif kind == "ordinal":
                        value = abs(x - y)
                    else:
                        value = float(x != y)
                    numerator += weights[c] * value
                    denominator += weights[c]
                row.append(numerator / denominator if denominator else 0)
            values.append(row)
        return np.array(values)

    def test_vectorized_matches_scalar_with_missing_and_weights(self):
        rng = np.random.default_rng(123)
        left = pd.DataFrame({
            "v": rng.uniform(-3, 3, 41),
            "cat": rng.choice(["a", "b", "c"], 41),
            "ord": rng.choice([0., .5, 1.], 41),
            "bin": rng.choice([0., 1.], 41),
        })
        right = left.sample(13, random_state=3).copy()
        weights = {"v": 1.3, "cat": .5, "ord": 2.1, "bin": 3.5}
        m = GowerMetric(["v"], ["cat"], {"ord": ["a", "b", "c"]},
                        ["bin"], weights).fit(left)
        # Values are already metric-coded when passed to distances.
        expected = self.scalar(left, right, m.ranges_, weights)
        np.testing.assert_allclose(m.distances(left, right), expected,
                                   atol=1e-14, rtol=1e-14)
        left.loc[[2, 5, 8], "cat"] = None
        left.loc[[3, 9], "v"] = np.nan
        right.iloc[[0, 4], right.columns.get_loc("bin")] = np.nan
        expected = self.scalar(left, right, m.ranges_, weights)
        np.testing.assert_allclose(m.distances(left, right), expected,
                                   atol=1e-14, rtol=1e-14)
        self.assertEqual(m.distances(left).shape, (41, 41))


if __name__ == "__main__":
    unittest.main()

class FastGowerTests(unittest.TestCase):
    def test_vectorized_backend_matches_stable_with_missing(self):
        rng = np.random.default_rng(9)
        frame = pd.DataFrame({
            "num": rng.normal(size=50),
            "cat": rng.choice(["a", "b", "c"], 50),
            "binary": rng.integers(0, 2, 50),
            "ordinal": rng.choice(["low", "mid", "high"], 50),
        })
        frame.loc[[5, 9], "num"] = np.nan
        frame.loc[[2, 8], "cat"] = None
        frame.loc[[1, 15], "ordinal"] = None
        schema = Schema(numeric=("num",), categorical=("cat",),
                        binary=("binary",),
                        ordinal={"ordinal": ("low", "mid", "high")})
        from clustelia import GeometrySpec
        from clustelia.geometry import ReferenceGeometry
        stable = ReferenceGeometry(
            GeometrySpec("gower", "none", missing="pairwise"), schema
        ).fit(frame)
        fast = ReferenceGeometry(
            GeometrySpec("gower", "none", missing="pairwise",
                         gower_backend="vectorized"), schema
        ).fit(frame)
        a, b = stable.transform(frame), fast.transform(frame)
        np.testing.assert_allclose(stable.distances(a), fast.distances(b),
                                   atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(stable.distances(a.iloc[:10], a.iloc[10:]),
                                   fast.distances(b.iloc[:10], b.iloc[10:]),
                                   atol=1e-13, rtol=1e-13)
        with self.assertRaisesRegex(ValueError, "gower_backend"):
            GeometrySpec("gower", "none", gower_backend="unknown")
        with self.assertRaisesRegex(ValueError, "requires"):
            GeometrySpec(gower_backend="vectorized")

class ProfileContrastTests(unittest.TestCase):
    def test_profile_has_baselines(self):
        rng = np.random.default_rng(2)
        x = pd.DataFrame({
            "value": np.r_[rng.normal(0, .2, 45),
                            rng.normal(10, .2, 45)],
            "segment": ["left"] * 45 + ["right"] * 45,
        })
        from clustelia import GeometrySpec
        result = cluster(
            x, effort="quick", identifiers=None,
            geometry=GeometrySpec(kind="gower", scaling="none"),
            families=("medoids",), k_range=(2,),
        )
        solution = result.select(result.candidate_ids[0])
        table = solution.profile_table()
        self.assertIn("median_delta", table.columns)
        self.assertIn("top_lift", table.columns)
        self.assertTrue(table.loc[table.type.eq("numeric"),
                                  "global_median"].notna().all())
        self.assertTrue(table.loc[table.type.eq("categorical"),
                                  "top_lift"].notna().all())
