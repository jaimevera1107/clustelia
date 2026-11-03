"""Regressions of the additional families, density and fuzzy objectives."""

import subprocess
import sys
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, pairwise_distances

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
    adapter_capabilities,
)
from clustelia.adapters import FittedCandidate
from clustelia.density import density_validity
from clustelia.geometry import ReferenceGeometry
from clustelia.optional import optional_version
from clustelia.search import stable_seed
from clustelia.selection import choose

HAS_DENSITY = optional_version("density") is not None
HAS_FUZZY = optional_version("fuzzy") is not None


class _BlobCase(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(
            n_samples=120, centers=3, cluster_std=0.25, random_state=18
        )
        self.policy = SelectionPolicy(
            subsamples=3, seed_repeats=2, shortlist_size=2
        )


class FamilyContracts(_BlobCase):
    def test_new_compact_families_preserve_labels_and_assignment(self):
        for name in ("minibatch", "birch", "bisecting", "spectral"):
            with self.subTest(name=name):
                model = AutoCluster(
                    candidates=(CandidateSpec(name, 3),),
                    selection=self.policy,
                ).fit(self.X)
                result = model.result_
                self.assertEqual(result.selection_status, "recommended")
                sol = result.select(result.recommendation.candidate_id)
                self.assertGreater(adjusted_rand_score(self.y, sol.labels_),
                                   0.95)
                if name == "spectral":
                    with self.assertRaises(CapabilityError):
                        sol.assign(self.X[:1])
                else:
                    np.testing.assert_array_equal(
                        sol.assign(self.X), sol.labels_
                    )
                row = model.results_.iloc[0]
                self.assertTrue(row.backend_parameters)
                if name == "birch":
                    self.assertIsNone(row.randomness)

    def test_ward_structure_is_conserved_and_owned(self):
        m = AutoCluster(
            candidates=(CandidateSpec("agglomerative", 3),),
            selection=self.policy,
        ).fit(self.X)
        sol = m.result_.select(m.results_.iloc[0].candidate_id)
        tree = sol.structure_
        self.assertEqual(tree["kind"], "ward_linkage")
        self.assertEqual(tree["children"].shape, (119, 2))
        tree["children"][:] = 0
        self.assertTrue(np.any(sol.structure_["children"] != 0))

    def test_optics_reachability_and_transductive_capability(self):
        m = AutoCluster(
            objective="density",
            candidates=(CandidateSpec("optics"),),
            selection=self.policy,
        ).fit(self.X)
        sol = m.result_.select(m.results_.iloc[0].candidate_id)
        structure = sol.structure_
        self.assertEqual(structure["kind"], "optics_reachability")
        self.assertEqual(set(structure["ordering"]), set(range(120)))
        self.assertEqual(len(structure["core_distances"]), 120)
        with self.assertRaises(CapabilityError):
            sol.assign(self.X[:2])

    def test_hdbscan_backend_and_strength_are_explicit(self):
        m = AutoCluster(
            objective="density",
            candidates=(
                CandidateSpec("hdbscan", min_samples=5, min_cluster_size=10),
            ),
            selection=self.policy,
        ).fit(self.X)
        sol = m.result_.select(m.results_.iloc[0].candidate_id)
        module = sol.pipeline_.estimator_.__class__.__module__
        self.assertTrue(module.startswith("sklearn"))
        membership = sol.membership_
        self.assertEqual(membership.semantics, "density_association_strength")
        self.assertEqual(membership.normalization,
                         "not_a_distribution_over_clusters")
        self.assertEqual(membership.values.shape, (120, 1))
        values = membership.values
        self.assertTrue(np.all((values >= 0) & (values <= 1)))

    def test_capability_registry_is_a_copy(self):
        caps = adapter_capabilities()
        self.assertEqual(len(caps), 15)
        self.assertIn("clara", caps)
        self.assertIn("includes the observation itself",
                      caps["hdbscan"]["min_samples_convention"])
        caps["kmeans"]["assign"] = False
        self.assertTrue(adapter_capabilities()["kmeans"]["assign"])

    def test_automatic_family_grid_uses_same_engine(self):
        m = AutoCluster(
            families=("minibatch", "birch", "bisecting"),
            k_range=(2, 3),
            selection=self.policy,
        ).fit(self.X)
        self.assertEqual(set(m.results_.algorithm),
                         {"minibatch", "birch", "bisecting"})
        self.assertEqual(len(m.results_), 6)
        self.assertEqual(m.result_.selection_status, "recommended")
        with self.assertRaises(ValueError):
            AutoCluster(
                families=("birch",), candidates=(CandidateSpec("birch", 3),)
            ).fit(self.X)
        with self.assertRaises(CapabilityError):
            AutoCluster(families=("hdbscan",)).fit(self.X)

    def test_all_failed_primary_validation_is_incomplete(self):
        original = FittedCandidate.fit

        def fail_validation(fitted, frame):
            if len(frame) == 96:
                raise RuntimeError("validation backend failed")
            return original(fitted, frame)

        with patch.object(FittedCandidate, "fit", fail_validation):
            m = AutoCluster(
                objective="probabilistic",
                probabilistic_metric="heldout_loglikelihood",
                candidates=(CandidateSpec("gmm", 3),),
                selection=self.policy,
            ).fit(self.X)
        self.assertEqual(m.result_.selection_status, "incomplete")
        diagnostics = m.results_.iloc[0].validation_diagnostics
        self.assertIn("validation backend failed", diagnostics["error"])

    def test_bayesian_bic_is_excluded_before_fitting(self):
        with patch.object(FittedCandidate, "fit") as spy:
            with self.assertRaisesRegex(CapabilityError,
                                        "no comparable BIC"):
                AutoCluster(
                    objective="probabilistic",
                    candidates=(CandidateSpec("bgmm", 3),),
                ).fit(self.X)
            spy.assert_not_called()

    def test_heldout_split_is_disjoint_and_budgeted(self):
        m = AutoCluster(
            objective="probabilistic",
            probabilistic_metric="heldout_loglikelihood",
            candidates=(CandidateSpec("gmm", 3), CandidateSpec("bgmm", 3)),
            selection=self.policy,
        ).fit(self.X)
        meta = m.result_.metadata
        validation = set(meta["validation_indices"])
        training = set(meta["training_indices"])
        self.assertFalse(validation & training)
        self.assertEqual(validation | training, set(range(120)))
        self.assertEqual(meta["fit_breakdown"]["exploration"], 2)
        self.assertEqual(meta["fit_breakdown"]["validation"], 2)
        self.assertEqual(sum(meta["fit_breakdown"].values()),
                         meta["fits_used"])
        for row in m.results_.itertuples():
            self.assertTrue(np.isfinite(row.heldout_loglikelihood))
            self.assertEqual(row.selection_metric, "heldout_loglikelihood")
            if row.algorithm == "bgmm":
                self.assertTrue(pd.isna(row.bic))

    def test_validation_preprocessing_never_sees_reserved_rows(self):
        records = []
        original = ReferenceGeometry.fit

        def capture(geometry, frame):
            records.append((list(frame.index), frame.to_numpy().copy()))
            return original(geometry, frame)

        with patch.object(ReferenceGeometry, "fit", capture):
            m = AutoCluster(
                objective="probabilistic",
                probabilistic_metric="heldout_loglikelihood",
                candidates=(CandidateSpec("gmm", 3),),
                selection=self.policy,
            ).fit(self.X)
        train = m.result_.metadata["training_indices"]
        validations = [v for indices, v in records if indices == train]
        self.assertEqual(len(validations), 1)
        np.testing.assert_array_equal(validations[0], self.X[train])
        # Independent reconstruction of the validation model and score.
        cid = m.results_.iloc[0].candidate_id
        validator = FittedCandidate(
            CandidateSpec("gmm", 3),
            GeometrySpec(),
            Schema(numeric=(0, 1)),
            stable_seed(42, cid, "validation"),
            256,
        ).fit(pd.DataFrame(self.X[train]))
        reserved = m.result_.metadata["validation_indices"]
        expected = validator.loglikelihood(pd.DataFrame(self.X[reserved]))
        self.assertAlmostEqual(m.results_.iloc[0].heldout_loglikelihood,
                               expected, places=12)

    def test_validation_fits_do_not_exceed_small_budget(self):
        m = AutoCluster(
            objective="probabilistic",
            probabilistic_metric="heldout_loglikelihood",
            candidates=(CandidateSpec("gmm", 3),),
            selection=self.policy,
            budget=SearchBudget(max_fits=2),
        ).fit(self.X)
        self.assertEqual(m.result_.metadata["fits_used"], 2)
        self.assertEqual(m.result_.selection_status, "incomplete")

    def test_failed_backend_warnings_are_retained(self):
        def fail(*args):
            warnings.warn("warning before backend failure", stacklevel=1)
            raise RuntimeError("fit failed")

        with patch.object(FittedCandidate, "fit", fail):
            with self.assertRaises(SearchFailedError) as ctx:
                AutoCluster(candidates=(CandidateSpec("kmeans", 3),)).fit(
                    self.X
                )
        table = ctx.exception.result.candidates_table()
        self.assertIn("warning before backend failure",
                      table.iloc[0].warnings)

    def test_core_import_does_not_import_optional_modules(self):
        code = (
            "import clustelia, sys; "
            "assert 'hdbscan' not in sys.modules; "
            "assert 'skfuzzy' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True,
                       capture_output=True)

    def test_invalid_new_parameters_fail_early(self):
        factories = (
            lambda: CandidateSpec("dbscan", k=3),
            lambda: CandidateSpec("fcm", 3, fuzzifier=1),
            lambda: CandidateSpec("spectral", 3, gamma=0),
            lambda: CandidateSpec("optics", xi=1),
            lambda: CandidateSpec("hdbscan", min_cluster_size=1),
            lambda: SelectionPolicy(density_tolerance=-1),
        )
        for factory in factories:
            with self.assertRaises(ValueError):
                factory()


class DensityContracts(_BlobCase):
    def test_missing_dbcv_is_explicit_and_has_no_substitute(self):
        with patch("clustelia.density.optional_version", return_value=None):
            m = AutoCluster(
                objective="density",
                candidates=(CandidateSpec("dbscan"),),
                selection=self.policy,
            ).fit(self.X)
        self.assertEqual(m.result_.selection_status, "incomplete")
        self.assertTrue(m.result_.metadata["incomplete_reasons"])
        self.assertIsNone(m.results_.iloc[0].primary_score)
        self.assertIn("clustelia[density]", m.results_.iloc[0].dbcv_reason)

    def test_gower_dbcv_is_not_assumed_valid(self):
        out = density_validity(pd.DataFrame(self.X), self.y, "gower", 256)
        self.assertTrue(out["primary_unavailable"])
        self.assertIsNone(out["dbcv"])

    @unittest.skipUnless(HAS_DENSITY, "requires density extra")
    def test_dbcv_matches_backend_and_is_label_invariant(self):
        from hdbscan.validity import validity_index

        expected = validity_index(self.X.astype(np.float64), self.y)
        actual = density_validity(self.X, self.y, "euclidean", 256)
        self.assertAlmostEqual(actual["dbcv"], expected, places=12)
        relabel = np.choose(self.y, [100, 9, 38])
        relabelled = density_validity(self.X, relabel, "euclidean", 256)
        self.assertAlmostEqual(relabelled["dbcv"], expected, places=12)

    @unittest.skipUnless(HAS_DENSITY, "requires density extra")
    def test_dbcv_noise_weighting_and_scale_invariance(self):
        base = density_validity(self.X, self.y, "euclidean", 256)["dbcv"]
        outliers = [[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]]
        noisy_X = np.vstack([self.X, outliers])
        noisy_y = np.r_[self.y, [-1, -1, -1]]
        score = density_validity(noisy_X, noisy_y, "euclidean", 256)["dbcv"]
        self.assertAlmostEqual(score, base * 120 / 123, places=12)
        scaled = density_validity(self.X * 10, self.y, "euclidean", 256)
        self.assertAlmostEqual(scaled["dbcv"], base, places=12)

    @unittest.skipUnless(HAS_DENSITY, "requires density extra")
    def test_density_confirmation_can_recommend(self):
        m = AutoCluster(
            objective="density",
            candidates=(CandidateSpec("dbscan"),),
            selection=self.policy,
        ).fit(self.X)
        self.assertEqual(m.result_.selection_status, "recommended")
        self.assertEqual(m.result_.metadata["fits_used"], 4)
        row = m.results_.iloc[0]
        self.assertEqual(row.selection_metric, "dbcv")
        self.assertEqual(row.dbcv_support, 120)
        self.assertEqual(row.common_support_count, 120)

    @unittest.skipUnless(HAS_DENSITY, "requires density extra")
    def test_dbcv_degeneracy_duplicates_and_memory_have_reasons(self):
        cases = (
            (self.X, np.zeros(120, dtype=int)),
            (np.r_[self.X, self.X[:1]], np.r_[self.y, self.y[:1]]),
        )
        for X, y in cases:
            out = density_validity(X, y, "euclidean", 256)
            self.assertIsNone(out["dbcv"])
            self.assertIsNotNone(out["dbcv_reason"])
        out = density_validity(self.X, self.y, "euclidean", 0.0001)
        self.assertIsNone(out["dbcv"])
        self.assertIn("MemoryBudgetError", out["dbcv_reason"])

    def test_density_tolerance_does_not_inherit_silhouette_tolerance(self):
        stats = {"median": 0.95, "evaluable_fraction": 1}
        rows = [
            {"candidate_id": cid, "primary_score": score,
             "composition": stats, "randomness": None, "confirmed": True,
             "coverage": 1.0, "fit_seconds": 1.0}
            for cid, score in [("a", 0.9), ("b", 0.89)]
        ]
        winner, _, admissible = choose(rows, "density", SelectionPolicy())
        self.assertIsNone(winner)
        self.assertEqual(admissible, ["a", "b"])
        strict = SelectionPolicy(density_tolerance=0)
        winner, _, _ = choose(rows, "density", strict)
        self.assertEqual(winner, "a")


@unittest.skipUnless(HAS_FUZZY, "requires fuzzy extra")
class FuzzyContracts(_BlobCase):
    def test_memberships_objective_and_rng_are_correct(self):
        from clustelia.fuzzy import FuzzyCMeans

        np.random.seed(342)
        before = np.random.get_state()
        model = FuzzyCMeans(3, 2.0, 42)
        labels = model.fit_predict(self.X)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_allclose(model.memberships_.sum(axis=1), 1)
        self.assertTrue(np.all(np.diff(model.objective_history_) <= 1e-9))
        d = pairwise_distances(self.X, model.cluster_centers_,
                               metric="sqeuclidean")
        expected = np.sum(model.memberships_**2 * d)
        self.assertAlmostEqual(model.objective_, expected, places=10)
        self.assertGreater(adjusted_rand_score(self.y, labels), 0.95)
        np.testing.assert_array_equal(model.predict(self.X), labels)

    def test_fuzzy_recommendation_semantics_and_assignment(self):
        m = AutoCluster(
            objective="fuzzy",
            candidates=(CandidateSpec("fcm", 3),),
            selection=self.policy,
        ).fit(self.X)
        self.assertEqual(m.result_.selection_status, "recommended")
        s = m.result_.select(m.result_.recommendation.candidate_id)
        self.assertEqual(s.membership_.semantics, "fuzzy_membership")
        np.testing.assert_array_equal(s.assign(self.X), s.labels_)
        diagnostics = m.results_.iloc[0].fuzzy_diagnostics
        self.assertGreater(diagnostics["fpc"], 1 / 3)

    def test_fuzzifier_and_representation_are_fixed_within_objective(self):
        groups = (
            (CandidateSpec("fcm", 2, fuzzifier=2),
             CandidateSpec("fcm", 3, fuzzifier=3)),
            (CandidateSpec("fcm", 2),
             CandidateSpec("fcm", 3, pca_variance=0.95)),
        )
        for specs in groups:
            with self.assertRaises(CapabilityError):
                AutoCluster(objective="fuzzy", candidates=specs).fit(self.X)

    def test_missing_fuzzy_dependency_excluded_before_fit(self):
        with patch("clustelia.planning.optional_version", return_value=None):
            with self.assertRaisesRegex(CapabilityError,
                                        "clustelia\\[fuzzy\\]"):
                AutoCluster(objective="fuzzy").fit(self.X)

    def test_multiple_admissible_fuzzy_solutions_are_ambiguous(self):
        policy = SelectionPolicy(
            subsamples=3, seed_repeats=2, shortlist_size=2, min_stability=0
        )
        m = AutoCluster(
            objective="fuzzy",
            candidates=(CandidateSpec("fcm", 2), CandidateSpec("fcm", 3)),
            selection=policy,
        ).fit(self.X)
        self.assertEqual(m.result_.selection_status, "ambiguous")
        self.assertIsNone(m.result_.recommendation)
        with self.assertRaises(NoRecommendationError):
            _ = m.labels_
        self.assertEqual(len(m.result_.metadata["admissible_candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
