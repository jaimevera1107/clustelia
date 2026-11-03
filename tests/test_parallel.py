"""Tests of the worker-process null-reference stage (SearchBudget.workers)."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from clustelia import (
    AutoCluster,
    CandidateSpec,
    GeometrySpec,
    Schema,
    SearchBudget,
    SelectionPolicy,
)
from clustelia.execution import (
    AUTO_MAX_WORKERS,
    AUTO_MIN_ROWS,
    FitBudgetExhaustedError,
    FitExecutor,
)

TIMING_TOKENS = ("second", "time", "elapsed", "duration")


def _policy():
    return SelectionPolicy(subsamples=3, seed_repeats=2, shortlist_size=3)


def _snapshot(result):
    """Everything that must not depend on the number of workers."""
    table = result.candidates_table()
    drop = [c for c in table.columns
            if any(token in c.lower() for token in TIMING_TOKENS)]
    meta = result.metadata
    return {
        "status": result.selection_status,
        "recommendation": (None if result.recommendation is None
                           else result.recommendation.candidate_id),
        "table": table.drop(columns=drop),
        "labels": {i: result.select(i).labels_.tolist()
                   for i in result.candidate_ids},
        "fits": (meta["fits_used"], dict(meta["fit_breakdown"])),
        "gap": meta["gap_table"],
        "selected_k": meta["selected_k"],
    }


def _assert_same(test, a, b):
    for key in ("status", "recommendation", "labels", "fits", "gap",
                "selected_k"):
        test.assertEqual(a[key], b[key], key)
    pd.testing.assert_frame_equal(a["table"], b["table"], check_exact=True)


def _fit(X, workers, **kwargs):
    budget = kwargs.pop("budget", SearchBudget())
    budget = SearchBudget(**{**budget.__dict__, "workers": workers})
    return AutoCluster(random_state=3, selection=_policy(), budget=budget,
                       **kwargs).fit(X).result_


def _length(labels):
    return len(labels)


class WorkerSelection(unittest.TestCase):
    def executor(self, workers):
        return FitExecutor(GeometrySpec(), Schema(numeric=("a", "b")),
                           SearchBudget(workers=workers), 0, 1)

    def test_budget_validation(self):
        for good in (None, 1, 2, 16, -1):
            self.assertEqual(SearchBudget(workers=good).workers, good)
        for bad in (0, -2, 1.5, "4"):
            with self.subTest(workers=bad), self.assertRaises(ValueError):
                SearchBudget(workers=bad)

    def test_auto_is_sequential_on_small_data(self):
        with patch("clustelia.execution.cpu_count", return_value=16):
            executor = self.executor(None)
            self.assertEqual(executor.workers_for(21, AUTO_MIN_ROWS - 1), 1)
            self.assertEqual(executor.workers_for(21, AUTO_MIN_ROWS),
                             AUTO_MAX_WORKERS)
            self.assertEqual(executor.workers_for(3, 10_000), 3)

    def test_auto_respects_the_cpu_count(self):
        with patch("clustelia.execution.cpu_count", return_value=2):
            self.assertEqual(self.executor(None).workers_for(21, 5000), 2)

    def test_explicit_values_ignore_the_row_threshold(self):
        with patch("clustelia.execution.cpu_count", return_value=16):
            self.assertEqual(self.executor(4).workers_for(21, 50), 4)
            self.assertEqual(self.executor(4).workers_for(2, 50), 2)
            self.assertEqual(self.executor(1).workers_for(21, 50_000), 1)
            self.assertEqual(self.executor(-1).workers_for(21, 50), 16)


class ParallelParity(unittest.TestCase):
    def test_numeric_search_is_identical(self):
        X, _ = make_blobs(240, centers=3, cluster_std=0.5, random_state=2)
        kwargs = {"families": ("kmeans", "agglomerative"),
                  "k_range": (2, 3, 4)}
        serial = _fit(X, 1, **kwargs)
        for workers in (2, 3):
            with self.subTest(workers=workers):
                parallel = _fit(X, workers, **kwargs)
                _assert_same(self, _snapshot(serial), _snapshot(parallel))
                self.assertEqual(parallel.metadata["parallel_workers"],
                                 workers)
        self.assertEqual(serial.metadata["parallel_workers"], 1)
        self.assertEqual(serial.selection_status, "recommended")

    def test_gower_search_is_identical(self):
        rng = np.random.default_rng(0)
        groups = np.repeat([0, 1, 2], 40)
        frame = pd.DataFrame({
            "x": rng.normal(groups * 6.0, 1.0),
            "c": np.array(["a", "b", "c"])[groups],
        })
        kwargs = {"schema": Schema(numeric=["x"], categorical=["c"]),
                  "k_range": (2, 3)}
        serial = _fit(frame, 1, **kwargs)
        parallel = _fit(frame, 2, **kwargs)
        self.assertEqual(parallel.metadata["structure_test"]["family"],
                         "permutation")
        _assert_same(self, _snapshot(serial), _snapshot(parallel))

    def test_failed_null_test_is_identical_and_exits_early(self):
        X = np.random.default_rng(0).uniform(size=(200, 2))
        kwargs = {"families": ("kmeans", "agglomerative"),
                  "k_range": (2, 3, 4)}
        serial = _fit(X, 1, **kwargs)
        parallel = _fit(X, 2, **kwargs)
        self.assertIsNone(parallel.recommendation)
        self.assertEqual(serial.metadata["fits_used"],
                         parallel.metadata["fits_used"])
        _assert_same(self, _snapshot(serial), _snapshot(parallel))

    def test_a_binding_budget_falls_back_to_the_sequential_path(self):
        X, _ = make_blobs(200, centers=3, cluster_std=0.5, random_state=2)
        kwargs = {"families": ("kmeans", "agglomerative"),
                  "k_range": (2, 3, 4),
                  "budget": SearchBudget(max_fits=14)}
        serial = _fit(X, 1, **kwargs)
        parallel = _fit(X, 2, **kwargs)
        self.assertEqual(serial.selection_status, "incomplete")
        self.assertEqual(serial.metadata["fits_used"], 14)
        _assert_same(self, _snapshot(serial), _snapshot(parallel))


class BatchSemantics(unittest.TestCase):
    def setUp(self):
        X, _ = make_blobs(60, centers=2, random_state=0)
        self.data = pd.DataFrame(X, columns=["a", "b"])
        self.schema = Schema(numeric=("a", "b"))

    def executor(self, workers, max_fits=100):
        return FitExecutor(GeometrySpec(), self.schema,
                           SearchBudget(workers=workers, max_fits=max_fits),
                           0, 1)

    def test_results_and_counts_match_the_sequential_path(self):
        specs = [CandidateSpec("kmeans", k) for k in (2, 3, 4)]
        counts = []
        for workers in (1, 2):
            executor = self.executor(workers)
            lengths = executor.fit_apply(specs, self.data, "p", "structure",
                                         _length)
            self.assertEqual(lengths, [60, 60, 60])
            counts.append((executor.fits_used, dict(executor.breakdown)))
        self.assertEqual(counts[0], counts[1])
        self.assertEqual(counts[0][0], 3)

    def test_first_error_in_task_order_later_fits_uncharged(self):
        # 500 clusters on 60 rows fails; the third fit must not be charged.
        specs = [CandidateSpec("kmeans", 2), CandidateSpec("kmeans", 500),
                 CandidateSpec("kmeans", 3)]
        for workers in (1, 2):
            with self.subTest(workers=workers):
                executor = self.executor(workers)
                with self.assertRaises(ValueError):
                    executor.fit_apply(specs, self.data, "p", "structure",
                                       _length)
                self.assertEqual(executor.fits_used, 2)
                self.assertEqual(executor.breakdown["structure"], 2)

    def test_exhausted_budget_stops_the_sequential_path(self):
        specs = [CandidateSpec("kmeans", k) for k in (2, 3, 4)]
        executor = self.executor(2, max_fits=2)
        with self.assertRaises(FitBudgetExhaustedError):
            executor.fit_apply(specs, self.data, "p", "structure", _length)
        self.assertEqual(executor.fits_used, 2)

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            self.executor(1).fit_apply([], self.data, "p", "nope", _length)


if __name__ == "__main__":
    unittest.main()
