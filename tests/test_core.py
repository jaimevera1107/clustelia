"""Contract and numerical regression tests of the core search."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler

from clustelia import (
    AutoCluster,
    CandidateSpec,
    CapabilityError,
    GeometrySpec,
    NoRecommendationError,
    Schema,
    SearchBudget,
    SearchFailedError,
    SelectionPolicy,
    load,
    save,
)
from clustelia.adapters import AlternateMedoids, FittedCandidate
from clustelia.evaluation import partition_metrics, stability_pair
from clustelia.execution import FitExecutor
from clustelia.geometry import ReferenceGeometry


class Contracts(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(
            n_samples=100, centers=3, cluster_std=0.25, random_state=18
        )
        # Fit-count contracts below describe the P0 replicas only.
        self.policy = SelectionPolicy(
            subsamples=3, seed_repeats=2, shortlist_size=2, null_replicates=0
        )

    def model(self, **kwargs):
        config = {
            "candidates": (CandidateSpec("kmeans", 3),),
            "selection": self.policy,
        }
        config.update(kwargs)
        return AutoCluster(**config)

    def test_recommendation_and_exact_pipeline(self):
        m = self.model().fit(self.X)
        self.assertEqual(m.result_.selection_status, "recommended")
        sol = m.result_.select(m.result_.recommendation.candidate_id)
        np.testing.assert_array_equal(sol.labels_, sol.assign(self.X))
        np.testing.assert_array_equal(m.labels_, m.best_estimator_.labels_)
        labels = sol.labels_
        labels[:] = -99
        self.assertTrue((sol.labels_ >= 0).all())
        self.assertEqual(m.result_.metadata["fits_used"], 6)

    def test_input_unchanged_and_external_y_ignored(self):
        df = pd.DataFrame(self.X, columns=["a", "b"])
        original = df.copy(deep=True)
        m = self.model().fit(df)
        before = m.result_.candidates_table()
        m.result_.external_evaluate(self.y)
        pd.testing.assert_frame_equal(before, m.result_.candidates_table())
        pd.testing.assert_frame_equal(df, original)
        with self.assertWarns(UserWarning):
            other = self.model().fit(df, y=np.arange(len(df)))
        np.testing.assert_array_equal(m.labels_, other.labels_)

    def test_common_reference_metric_with_pca(self):
        X = np.column_stack([self.X, self.X[:, 0] * 500 + np.arange(100)])
        specs = (
            CandidateSpec("kmeans", 3),
            CandidateSpec("kmeans", 3, pca_variance=0.8),
        )
        m = self.model(candidates=specs).fit(X)
        ref = StandardScaler().fit_transform(X)
        ids = m.result_.metadata["metric_indices"]
        for _, row in m.results_.iterrows():
            labels = m.result_.select(row.candidate_id).labels_[ids]
            expected = silhouette_score(ref[ids], labels)
            self.assertAlmostEqual(row.silhouette, expected, places=12)
            if row.parameters["pca_variance"]:
                pipeline = m.result_.select(row.candidate_id).pipeline_
                np.testing.assert_allclose(
                    pipeline.reducer_.explained_variance_ratio_,
                    PCA(0.8, svd_solver="full")
                    .fit(ref)
                    .explained_variance_ratio_,
                )

    def test_seed_order_independence(self):
        a, b = CandidateSpec("kmeans", 2), CandidateSpec("kmeans", 3)
        left = self.model(candidates=(a, b)).fit(self.X)
        right = self.model(candidates=(b, a)).fit(self.X)
        for cid in left.results_.candidate_id:
            np.testing.assert_array_equal(
                left.result_.select(cid).labels_,
                right.result_.select(cid).labels_,
            )

    def test_clone(self):
        m = self.model(schema=Schema(numeric=(0, 1)))
        c = clone(m)
        self.assertIsNot(c, m)
        self.assertFalse(hasattr(c, "result_"))
        c.fit(self.X)

    def test_prediction_schema_and_one_row(self):
        df = pd.DataFrame(self.X, columns=["a", "b"])
        m = self.model().fit(df)
        s = m.result_.select(m.result_.recommendation.candidate_id)
        self.assertEqual(s.assign(df.iloc[:1]).shape, (1,))
        for invalid in (df[["b", "a"]], df.assign(extra=1), df[["a"]]):
            with self.assertRaises(ValueError):
                s.assign(invalid)

    def test_invalid_inputs_and_no_silent_densification(self):
        invalid_inputs = (
            self.X[:, 0],
            np.full((10, 2), np.nan),
            np.full((10, 2), np.inf),
            np.ones((10, 2)),
        )
        for X in invalid_inputs:
            with self.assertRaises(ValueError):
                self.model().fit(X)
        with self.assertRaises(CapabilityError):
            self.model().fit(csr_matrix(self.X))
        df = pd.DataFrame(self.X)
        df.index = [0] * len(df)
        with self.assertRaises(ValueError):
            self.model().fit(df)

    def test_refit_clears_stale_success(self):
        m = self.model().fit(self.X)
        with self.assertRaises(ValueError):
            m.fit(np.ones((100, 2)))
        self.assertFalse(hasattr(m, "result_"))

    def test_budget_includes_confirmation(self):
        m = self.model(budget=SearchBudget(max_fits=2)).fit(self.X)
        self.assertEqual(m.result_.selection_status, "incomplete")
        self.assertEqual(m.result_.metadata["fits_used"], 2)
        self.assertIsNone(m.result_.recommendation)
        with self.assertRaises(NoRecommendationError):
            _ = m.labels_

    def test_max_candidates_is_incomplete(self):
        specs = (CandidateSpec("kmeans", 2), CandidateSpec("kmeans", 3))
        m = self.model(
            candidates=specs, budget=SearchBudget(max_candidates=1)
        ).fit(self.X)
        self.assertEqual(m.result_.selection_status, "incomplete")

    def test_timeout_no_false_recommendation(self):
        # A fake clock keeps the test independent of the OS timer resolution
        # (Windows advances in ~15 ms ticks, so a tiny real timeout can
        # still see zero elapsed time and allow a first fit).
        with patch.object(FitExecutor, "elapsed", return_value=10.0):
            m = self.model(budget=SearchBudget(timeout_s=1.0)).fit(self.X)
        self.assertEqual(m.result_.selection_status, "incomplete")
        self.assertEqual(m.result_.metadata["fits_used"], 0)

    def test_error_record_and_raise(self):
        failure = RuntimeError("backend failed")
        with patch.object(FittedCandidate, "fit", side_effect=failure):
            with self.assertRaises(SearchFailedError) as ctx:
                self.model().fit(self.X)
            table = ctx.exception.result.candidates_table()
            self.assertIn("backend failed", table.iloc[0].error)
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                self.model(error_policy="raise").fit(self.X)

    def test_failure_does_not_abort_other_candidate(self):
        original = FittedCandidate.fit

        def selectively(fitted, frame):
            if fitted.spec.k == 2:
                raise ValueError("deliberate adapter failure")
            return original(fitted, frame)

        with patch.object(FittedCandidate, "fit", selectively):
            m = self.model(
                candidates=(
                    CandidateSpec("kmeans", 2),
                    CandidateSpec("kmeans", 3),
                )
            ).fit(self.X)
        self.assertEqual(m.results_.iloc[0].status, "failed")
        self.assertEqual(m.result_.selection_status, "recommended")

    def test_every_replica_refits_geometry(self):
        original = ReferenceGeometry.fit
        lengths = []

        def track(geometry, frame):
            lengths.append(len(frame))
            return original(geometry, frame)

        with patch.object(ReferenceGeometry, "fit", track):
            self.model().fit(self.X)
        self.assertEqual(lengths.count(80), 3)
        # Reference, exploration and two seed replicas.
        self.assertEqual(lengths.count(100), 4)

    def test_degenerate_stability_is_not_one(self):
        for labels in (np.zeros(100), np.arange(100)):
            self.assertIsNone(stability_pair(labels, labels, 20)["ari"])
        alternating = np.arange(6) % 2
        self.assertIsNone(stability_pair(alternating, alternating, 20)["ari"])

    def test_metrics_noise_and_label_invariance(self):
        labels = np.array([0, 0, 0, 1, 1, 1, -1])
        X = np.array([[0], [0.1], [0.2], [4], [4.1], [4.2], [20]])
        D = pairwise_distances(X)
        a = partition_metrics(labels, D, np.arange(7))
        relabelled = np.where(labels == 0, 50, np.where(labels == 1, 12, -1))
        b = partition_metrics(relabelled, D, np.arange(7))
        self.assertEqual(a, b)
        self.assertEqual(a["n_clusters"], 2)
        self.assertEqual(a["noise_count"], 1)
        self.assertAlmostEqual(
            a["silhouette"], silhouette_score(X[:6], labels[:6])
        )
        all_noise = partition_metrics(np.full(7, -1), D, np.arange(7))
        self.assertIsNone(all_noise["silhouette"])

    def test_density_is_exploratory_and_transductive(self):
        with patch("clustelia.density.optional_version", return_value=None):
            m = AutoCluster(
                objective="density",
                candidates=(CandidateSpec("dbscan", eps=0.1),),
                selection=self.policy,
            ).fit(self.X)
        self.assertEqual(m.result_.selection_status, "incomplete")
        s = m.result_.select(m.results_.iloc[0].candidate_id)
        with self.assertRaises(CapabilityError):
            s.assign(self.X[:1])

    def test_agglomerative_transductive(self):
        m = self.model(
            candidates=(CandidateSpec("agglomerative", 3),)
        ).fit(self.X)
        s = m.result_.select(m.results_.iloc[0].candidate_id)
        self.assertIsNone(m.results_.iloc[0].randomness)
        with self.assertRaises(CapabilityError):
            s.assign(self.X[:1])

    def test_small_sample_no_false_stability(self):
        m = self.model().fit(self.X[:10])
        self.assertEqual(
            m.result_.selection_status, "no_admissible_solution"
        )
        with self.assertRaises(NoRecommendationError):
            self.model().fit_predict(self.X[:10])

    def test_gmm_membership_and_comparable_bic(self):
        m = AutoCluster(
            objective="probabilistic", k_range=(2, 3), selection=self.policy
        ).fit(self.X)
        s = m.result_.select(m.result_.recommendation.candidate_id)
        np.testing.assert_allclose(s.membership_.values.sum(axis=1), 1)
        self.assertEqual(s.membership_.semantics, "mixture_posterior")
        with self.assertRaises(CapabilityError):
            AutoCluster(
                objective="probabilistic",
                candidates=(
                    CandidateSpec("gmm", 2),
                    CandidateSpec("gmm", 2, pca_variance=0.9),
                ),
            ).fit(self.X)

    def test_gower_hand_computed_and_equal_original_weights(self):
        df = pd.DataFrame(
            {"x": [0.0, 5.0, 10.0, 0.0], "c": ["a", "b", "a", "a"],
             "const": 1}
        )
        schema = Schema(numeric=("x", "const"), categorical=("c",))
        geom = ReferenceGeometry(GeometrySpec("gower", "none"), schema)
        geom.fit(df)
        D = geom.distances(geom.transform(df))
        self.assertAlmostEqual(D[0, 1], 0.75)
        self.assertAlmostEqual(D[0, 2], 0.5)
        self.assertEqual(geom.columns_, ["x", "c"])

    def test_mixed_schema_and_unknown_categories(self):
        df = pd.DataFrame({
            "x": self.X[:, 0],
            "c": np.where(self.y == 0, "A", "B"),
            "id": np.arange(100),
        })
        schema = Schema(numeric=["x"], categorical=["c"], identifiers=["id"])
        m = self.model(
            schema=schema, candidates=(CandidateSpec("medoids", 3),)
        ).fit(df)
        sol = m.result_.select(m.results_.iloc[0].candidate_id)
        np.testing.assert_array_equal(sol.labels_, sol.assign(df))
        new = df.iloc[:1].copy()
        new["c"] = "unseen"
        with self.assertRaises(ValueError):
            sol.assign(new)
        diagnostics = m.result_.metadata["diagnostics"]
        self.assertEqual(diagnostics["excluded_identifiers"], ["id"])

    def test_memory_guard(self):
        budget = SearchBudget(max_pairwise_mb=0.0001)
        with self.assertRaises(MemoryError):
            self.model(budget=budget).fit(self.X)

    def test_capability_exclusions(self):
        m = self.model(
            candidates=(CandidateSpec("gmm", 3), CandidateSpec("kmeans", 3))
        ).fit(self.X)
        self.assertEqual(len(m.result_.metadata["exclusions"]), 1)
        with self.assertRaises(CapabilityError):
            self.model().fit(self.X, sample_weight=np.ones(100))

    def test_save_load_and_trust_boundary(self):
        m = self.model().fit(self.X)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            save(m, path)
            with self.assertRaises(ValueError):
                load(path)
            restored = load(path, trusted=True)
            np.testing.assert_array_equal(restored.labels_, m.labels_)
            np.testing.assert_array_equal(
                restored.best_estimator_.assign(pd.DataFrame(self.X)),
                m.labels_,
            )

    def test_medoids_is_local_descent(self):
        D = pairwise_distances(self.X[:30])
        seed = 42
        indices = np.random.default_rng(seed).choice(30, 3, replace=False)
        initial = D[:, indices].min(axis=1).sum()
        m = AlternateMedoids(3, seed)
        m.fit_predict(D)
        self.assertLessEqual(m.inertia_, initial + 1e-10)
        self.assertTrue(m.converged_)

    def test_configs_reject_invalid_values(self):
        factories = (
            lambda: SearchBudget(max_fits=0),
            lambda: SelectionPolicy(subsamples=0),
            lambda: SelectionPolicy(min_stability=float("nan")),
            lambda: GeometrySpec("cosine"),
            lambda: Schema(numeric=["a"], categorical=["a"]),
            lambda: CandidateSpec("kmeans", 1),
        )
        for factory in factories:
            with self.assertRaises(ValueError):
                factory()


if __name__ == "__main__":
    unittest.main()
