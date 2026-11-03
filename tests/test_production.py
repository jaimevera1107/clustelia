"""Regression tests for the 0.1.0a3 hardening changes."""

import os
import pickle
import platform
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

import clustelia
from clustelia import (
    AutoCluster,
    CandidateSpec,
    ClusteliaError,
    MemoryBudgetError,
    NoRecommendationError,
    Schema,
    SearchBudget,
    SelectionPolicy,
    load,
    save,
)
from clustelia._version import __version__ as source_version
from clustelia.medoids import AlternateMedoids
from clustelia.selection import choose


def _fast_policy(**overrides):
    config = {"subsamples": 3, "seed_repeats": 2, "shortlist_size": 2}
    config.update(overrides)
    return SelectionPolicy(**config)


def _stable_table(model):
    table = model.result_.candidates_table()
    return table.drop(columns=["fit_seconds"])


class InputNormalisation(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=100, centers=3, cluster_std=0.25, random_state=18
        )

    def test_numpy_integers_are_accepted_and_normalised(self):
        spec = CandidateSpec("kmeans", np.int64(3), min_samples=np.int32(4))
        self.assertIs(type(spec.k), int)
        self.assertIs(type(spec.min_samples), int)
        self.assertEqual(spec, CandidateSpec("kmeans", 3, min_samples=4))
        model = AutoCluster(
            k_range=np.arange(2, 4), selection=_fast_policy(),
            random_state=np.int64(7),
        ).fit(self.X)
        self.assertEqual(len(model.results_), 6)

    def test_booleans_are_not_integers(self):
        with self.assertRaises(ValueError):
            CandidateSpec("kmeans", True)
        with self.assertRaises(ValueError):
            SearchBudget(max_fits=True)

    def test_invalid_k_range_fails_with_clear_message(self):
        for k_range in ((), (1, 2), (2.5,), "23"):
            with self.subTest(k_range=k_range):
                with self.assertRaisesRegex(ValueError, "k_range"):
                    AutoCluster(k_range=k_range).fit(self.X)

    def test_schema_rejects_bare_strings(self):
        with self.assertRaisesRegex(ValueError, "single string"):
            Schema(numeric="age")


class ExceptionContracts(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=100, centers=3, cluster_std=0.25, random_state=18
        )

    def test_hasattr_is_false_without_recommendation(self):
        model = AutoCluster(
            candidates=(CandidateSpec("kmeans", 3),), selection=_fast_policy(),
        ).fit(self.X[:10])
        self.assertIsNone(model.result_.recommendation)
        self.assertFalse(hasattr(model, "labels_"))
        self.assertFalse(hasattr(model, "best_estimator_"))
        with self.assertRaises(NoRecommendationError):
            _ = model.labels_

    def test_memory_budget_error_hierarchy(self):
        self.assertTrue(issubclass(MemoryBudgetError, MemoryError))
        self.assertTrue(issubclass(MemoryBudgetError, ClusteliaError))
        with self.assertRaises(MemoryBudgetError):
            AutoCluster(
                candidates=(CandidateSpec("kmeans", 3),),
                budget=SearchBudget(max_pairwise_mb=0.0001),
            ).fit(self.X)

    def test_failed_fit_sets_no_fitted_attribute(self):
        model = AutoCluster(
            candidates=(CandidateSpec("kmeans", 3),),
            budget=SearchBudget(max_pairwise_mb=0.0001),
        )
        with self.assertRaises(MemoryError):
            model.fit(self.X)
        for name in ("result_", "results_", "n_features_in_"):
            self.assertFalse(hasattr(model, name), name)


class Reproducibility(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=120, centers=3, cluster_std=0.6, random_state=3
        )

    def test_repeated_runs_are_identical(self):
        def run():
            return AutoCluster(
                k_range=(2, 3, 4), selection=_fast_policy()
            ).fit(self.X)

        first, second = run(), run()
        pd.testing.assert_frame_equal(_stable_table(first),
                                      _stable_table(second))
        self.assertEqual(first.result_.recommendation,
                         second.result_.recommendation)

    def test_default_tiebreak_ignores_wall_clock(self):
        stats = {"median": 0.9, "evaluable_fraction": 1.0}
        rows = [
            {"candidate_id": "a", "primary_score": 0.5, "composition": stats,
             "randomness": None, "confirmed": True, "coverage": 1.0,
             "fit_seconds": 9.0, "fits_used": 11},
            {"candidate_id": "b", "primary_score": 0.5, "composition": stats,
             "randomness": None, "confirmed": True, "coverage": 1.0,
             "fit_seconds": 0.1, "fits_used": 14},
        ]
        winner, _, _ = choose(rows, "compact", SelectionPolicy())
        self.assertEqual(winner, "a")
        legacy = SelectionPolicy(cost_tiebreak="fit_seconds")
        winner, _, _ = choose(rows, "compact", legacy)
        self.assertEqual(winner, "b")

    def test_fits_used_per_candidate_adds_up(self):
        model = AutoCluster(
            k_range=(2, 3), selection=_fast_policy()
        ).fit(self.X)
        meta = model.result_.metadata
        self.assertEqual(model.results_["fits_used"].sum(), meta["fits_used"])


class MemoryFootprint(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=100, centers=3, cluster_std=0.25, random_state=18
        )

    def test_solutions_share_one_training_table(self):
        model = AutoCluster(
            k_range=(2, 3), selection=_fast_policy()
        ).fit(self.X)
        frames = {
            id(model.result_.select(cid)._frame)
            for cid in model.result_.candidate_ids
        }
        self.assertEqual(len(frames), 1)

    def test_medoids_keep_only_prototypes(self):
        model = AutoCluster(
            candidates=(CandidateSpec("medoids", 3),), selection=_fast_policy()
        ).fit(self.X)
        solution = model.result_.select(model.result_.candidate_ids[0])
        self.assertEqual(len(solution.pipeline_.prototypes_), 3)
        np.testing.assert_array_equal(solution.assign(self.X),
                                      solution.labels_)

    def test_medoids_validate_distance_matrix(self):
        with self.assertRaises(ValueError):
            AlternateMedoids(2, 0).fit_predict(np.zeros((3, 2)))


class Persistence(unittest.TestCase):
    def setUp(self):
        X, _ = make_blobs(
            n_samples=100, centers=3, cluster_std=0.25, random_state=18
        )
        self.model = AutoCluster(
            candidates=(CandidateSpec("kmeans", 3),), selection=_fast_policy()
        ).fit(X)

    def test_save_is_atomic_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            save(self.model, path)
            self.assertEqual(os.listdir(directory), ["model.pkl"])
            restored = load(path, trusted=True)
            np.testing.assert_array_equal(restored.labels_, self.model.labels_)

    def test_save_rejects_unfitted_and_foreign_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            with self.assertRaises(ValueError):
                save(AutoCluster(), path)
            with self.assertRaises(TypeError):
                save(object(), path)
            self.assertFalse(path.exists())

    def test_load_rejects_non_envelope_and_mismatched_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "other.pkl"
            path.write_bytes(pickle.dumps([1, 2, 3]))
            with self.assertRaisesRegex(ValueError, "envelope"):
                load(path, trusted=True)
            save(self.model, path)
            envelope = pickle.loads(path.read_bytes())
            envelope["sklearn"] = "0.0"
            path.write_bytes(pickle.dumps(envelope))
            with self.assertRaisesRegex(ValueError, "sklearn"):
                load(path, trusted=True)

    def test_envelope_records_python_series_and_checks_it(self):
        # The envelope format (major, minor) strings must stay stable so
        # files written by earlier 0.2 releases keep loading.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            save(self.model, path)
            envelope = pickle.loads(path.read_bytes())
            major, minor, _ = platform.python_version_tuple()
            self.assertEqual(envelope["python"], (major, minor))
            envelope["python"] = ("2", "7")
            path.write_bytes(pickle.dumps(envelope))
            with self.assertRaisesRegex(ValueError, "python"):
                load(path, trusted=True)


class ResultContracts(unittest.TestCase):
    def test_external_evaluate_validates_length_once(self):
        X, y = make_blobs(
            n_samples=100, centers=3, cluster_std=0.25, random_state=18
        )
        model = AutoCluster(
            candidates=(CandidateSpec("kmeans", 3),), selection=_fast_policy()
        ).fit(X)
        with self.assertRaises(ValueError):
            model.result_.external_evaluate(y[:-1])
        table = model.result_.external_evaluate(y)
        self.assertAlmostEqual(table.iloc[0].ari, 1.0)

    def test_public_api_and_single_version(self):
        self.assertEqual(clustelia.__version__, source_version)
        for name in clustelia.__all__:
            self.assertTrue(hasattr(clustelia, name), name)


if __name__ == "__main__":
    unittest.main()


class MedoidRestarts(unittest.TestCase):
    def setUp(self):
        X, _ = make_blobs(n_samples=90, centers=4, cluster_std=1.5,
                          random_state=3)
        diff = X[:, None, :] - X[None, :, :]
        self.D = np.sqrt((diff ** 2).sum(-1))

    def test_single_restart_reproduces_one_run(self):
        one = AlternateMedoids(4, 11).fit_predict(self.D)
        again = AlternateMedoids(4, 11, n_init=1).fit_predict(self.D)
        np.testing.assert_array_equal(one, again)

    def test_restarts_never_increase_cost(self):
        for seed in range(5):
            single = AlternateMedoids(4, seed, n_init=1)
            single.fit_predict(self.D)
            multi = AlternateMedoids(4, seed, n_init=5)
            multi.fit_predict(self.D)
            self.assertLessEqual(multi.inertia_, single.inertia_ + 1e-12)

    def test_restarts_are_validated(self):
        with self.assertRaises(ValueError):
            AlternateMedoids(3, 0, n_init=0)
