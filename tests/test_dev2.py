"""Focused regressions for structural edge cases and production utilities."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from clustelia import AutoCluster, CandidateSpec, SelectionPolicy, load
from clustelia.config import GeometrySpec, Schema
from clustelia.gap import gap_table
from clustelia.geometry import GowerMetric
from clustelia.structure import generate_null


class GapDegeneracies(unittest.TestCase):
    def test_exact_zero_with_positive_references_is_positive_infinity(self):
        row = {"candidate_id": "perfect", "status": "fitted",
               "within_dispersion": 0., "parameters": {"k": 2}}
        entry = gap_table([row], {"perfect": [1., 1.2, 2.]})[0]
        self.assertTrue(np.isposinf(entry.gap))
        self.assertTrue(np.isfinite(entry.standard_error))

    def test_negative_and_zero_null_is_not_a_finite_gap(self):
        row = {"candidate_id": "x", "status": "fitted",
               "within_dispersion": 0., "parameters": {"k": 2}}
        self.assertEqual(gap_table([row], {"x": [0., 1.]}), [])
        row["within_dispersion"] = -1.
        self.assertEqual(gap_table([row], {"x": [1., 1.]}), [])


class GowerComparability(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"cat": ["a", None, "a", None],
                                   "binary": [0, 0, 1, 0]})

    def metric(self, policy="error", *, vectorized=False):
        return GowerMetric([], ["cat"], binary=["binary"],
                           incomparable=policy, vectorized=vectorized).fit(
                               self.frame
                           )

    def test_incomparable_pairs_are_not_perfect_matches(self):
        metric = self.metric()
        with self.assertRaisesRegex(ValueError, "comparable"):
            metric.distances(metric.transform(self.frame))
        metric = self.metric("max")
        coded = metric.transform(self.frame)
        D = metric.distances(coded)
        self.assertEqual(D[1, 3], 1.)
        self.assertEqual(D[1, 1], 0.)
        self.assertEqual(metric.comparable_share(coded)[1, 3], 0.)
        legacy = self.metric("zero")
        self.assertEqual(legacy.distances(coded)[1, 3], 0.)

    def test_row_blocks_same_as_dense(self):
        for backend in (False, True):
            metric = self.metric("max", vectorized=backend)
            coded = metric.transform(self.frame)
            dense = metric.distances(coded)
            blocks = list(metric.distance_blocks(coded, block_rows=2))
            self.assertEqual([(a, b) for a,b,_ in blocks], [(0,2),(2,4)])
            np.testing.assert_allclose(np.vstack([b for _,_,b in blocks]),
                                       dense, rtol=0, atol=1e-14)

    def test_low_comparability_policy(self):
        metric = GowerMetric([], ["cat"], binary=["binary"],
                             incomparable="max", min_comparable_share=.75)
        coded = metric.fit(self.frame).transform(self.frame)
        self.assertEqual(metric.distances(coded)[1, 3], 1.)
        same = pd.DataFrame({"cat": ["a", "a", "b"],
                             "binary": [0, 0, 1]})
        m2 = GowerMetric([], ["cat"], binary=["binary"],
                         incomparable="max", min_comparable_share=.75)
        code2 = m2.fit(same).transform(same)
        self.assertEqual(m2.distances(code2)[0, 1], 1.)
        with self.assertRaises(ValueError):
            GeometrySpec(kind="euclidean", null_blocks=(("cat",),))


class PermutationBlockTests(unittest.TestCase):
    def test_preserves_dummy_invariants(self):
        frame = pd.DataFrame({"a": [0,1,0,1,0,1],
                              "b": [1,0,1,0,1,0],
                              "category": list("aabbcc")})
        schema = Schema(numeric=("a","b"), categorical=("category",))
        generated = generate_null("permutation", frame, schema,
                                  np.random.default_rng(4),
                                  blocks=(("a", "b"),))
        self.assertTrue((generated["a"] + generated["b"] == 1).all())
        with self.assertRaisesRegex(ValueError, "features"):
            generate_null("permutation", frame, schema,
                          np.random.default_rng(1), blocks=(("fake",),))


class UserExperience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(4)
        X = np.r_[rng.normal(-5, .1, (12,2)),
                  rng.normal(5, .1, (12,2))]
        cls.result = AutoCluster(
            candidates=(CandidateSpec("kmeans", k=2),),
            selection=SelectionPolicy.p0(subsamples=2, seed_repeats=1),
            random_state=1,
        ).fit(pd.DataFrame(X, columns=["x", "y"])).result_

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "result.clustelia"
            self.result.save(file)
            with self.assertRaises(ValueError):
                load(file)
            other = load(file, trusted=True)
            self.assertEqual(other.summary()["status"],
                             self.result.summary()["status"])
            np.testing.assert_array_equal(
                other.select(other.candidate_ids[0]).labels_,
                self.result.select(self.result.candidate_ids[0]).labels_)

    def test_profiles_and_exports(self):
        result = self.result
        sol = result.select(result.candidate_ids[0])
        self.assertEqual(len(sol.feature_importance_table()), 2)
        self.assertEqual(sol.diagnostics_table().iloc[0].n_rows, 24)
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in (".csv", ".md", ".html", ".json"):
                path = result.export_report(Path(tmp) / f"result{suffix}")
                self.assertGreater(path.stat().st_size, 100)
            payload = json.loads((Path(tmp) / "result.json").read_text())
            self.assertEqual(payload["summary"]["rows"], 24)
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            return
        self.assertIsNotNone(sol.plot_importance())
        self.assertIsNotNone(sol.plot_heatmap())
        self.assertIsNotNone(sol.plot_projection())


if __name__ == "__main__":
    unittest.main()

class SampledMedoidRegression(unittest.TestCase):
    def test_euclidean_reproducible_and_assignable(self):
        from sklearn.metrics import adjusted_rand_score
        rng = np.random.default_rng(15)
        X = np.r_[rng.normal(-3,.1,(35,2)), rng.normal(3,.1,(35,2))]
        spec = CandidateSpec("clara", k=2, clara_sample_size=24,
                             clara_samples=3)
        options = {"candidates": (spec,),
                   "selection": SelectionPolicy.p0(
                       subsamples=2, seed_repeats=1
                   ), "random_state": 3}
        left = AutoCluster(**options).fit(X)
        right = AutoCluster(**options).fit(X)
        np.testing.assert_array_equal(left.result_.candidate_ids,
                                      right.result_.candidate_ids)
        first = left.result_.select(left.result_.candidate_ids[0])
        second = right.result_.select(right.result_.candidate_ids[0])
        np.testing.assert_array_equal(first.labels_, second.labels_)
        self.assertGreater(adjusted_rand_score(
            [0]*35+[1]*35, first.labels_
        ), .9)
        self.assertEqual(len(first.assign(X[:5])), 5)

    def test_mixed_gower(self):
        values = pd.DataFrame({
            "num": [0.1,0.2,0.3,0.4,10,10.2,10.3,10.4]*4,
            "category": (["a"]*4 + ["b"]*4)*4,
        })
        model = AutoCluster(
            schema=Schema(numeric=["num"], categorical=["category"]),
            geometry=GeometrySpec(kind="gower", scaling="none"),
            candidates=(CandidateSpec("clara", k=2,
                        clara_sample_size=15, clara_samples=2),),
            selection=SelectionPolicy.p0(subsamples=2, seed_repeats=1),
            random_state=7,
        ).fit(values)
        self.assertEqual(model.result_.selection_status, "recommended")
        solution = model.result_.select(model.result_.candidate_ids[0])
        self.assertEqual(len(np.unique(solution.labels_)), 2)
        self.assertEqual(len(solution.assign(values.iloc[:3])), 3)
