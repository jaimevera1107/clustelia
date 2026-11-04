"""Regression tests for audit findings in selection and result reporting."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from clustelia import AutoCluster, Schema, SelectionPolicy, cluster
from clustelia.gap import GapEntry
from clustelia.results import ExperimentResult, Solution
from clustelia.structure import StructureTester


class AuditRegressions(unittest.TestCase):
    """Prevent regressions in the audited public and internal contracts."""

    def setUp(self):
        self.X, _ = make_blobs(
            n_samples=120, centers=3, cluster_std=0.35,
            random_state=17,
        )

    def test_quick_default_search_is_not_truncated(self):
        result = cluster(self.X, effort="quick", random_state=3)
        self.assertEqual(result.selection_status, "recommended")
        self.assertFalse(any(
            e["reason"] == "max_candidates budget"
            for e in result.metadata["exclusions"]
        ))
        self.assertEqual(result.summary()["effort"], "quick")
        self.assertEqual(result.summary()["validation_scope"],
                         "reference_screen_and_stability")

    def test_null_backend_failure_is_incomplete_in_both_policies(self):
        for k_rule in ("gap", "silhouette"):
            with self.subTest(k_rule=k_rule):
                policy = SelectionPolicy(
                    null_replicates=2, k_selection=k_rule,
                    subsamples=2, seed_repeats=1,
                )
                with patch.object(
                    StructureTester, "_null_scores",
                    side_effect=ValueError("synthetic backend failure"),
                ):
                    result = AutoCluster(
                        families=("kmeans",), k_range=(2, 3),
                        selection=policy, random_state=3,
                    ).fit(self.X).result_
                self.assertEqual(result.selection_status, "incomplete")
                self.assertIsNone(result.recommendation)
                self.assertTrue(result.metadata["structure_test"]["failed"])
                self.assertIn(
                    "null reference fit failed",
                    result.metadata["incomplete_reasons"][0],
                )

    def test_missing_gap_for_any_observed_k_is_incomplete(self):
        policy = SelectionPolicy(
            null_replicates=2, subsamples=2, seed_repeats=1,
        )
        for table in ([], [GapEntry(2, "not-used", 0.2, 0.1)]):
            with self.subTest(table=table):
                with patch("clustelia.search.gap_table", return_value=table):
                    result = AutoCluster(
                        families=("kmeans",), k_range=(2, 3),
                        selection=policy, random_state=3,
                    ).fit(self.X).result_
                self.assertEqual(result.selection_status, "incomplete")
                self.assertIsNone(result.recommendation)
                self.assertIsNone(result.metadata["selected_k"])
                self.assertIn(
                    "gap statistic unavailable",
                    result.metadata["incomplete_reasons"][0],
                )

    def test_ranking_uses_primary_score_for_every_objective(self):
        scores = (("a", 0.1), ("b", 0.8), ("c", 0.4))
        rows = [
            {
                "candidate_id": identifier, "algorithm": "example",
                "status": "fitted", "confirmed": True,
                "primary_score": value, "silhouette": 1.0 - value,
                "n_clusters": 2, "coverage": 1.0,
                "rejection_reasons": [],
            }
            for identifier, value in scores
        ]
        for objective in ("compact", "probabilistic", "density", "fuzzy"):
            with self.subTest(objective=objective):
                result = ExperimentResult(
                    "ambiguous", rows, {}, None, {"objective": objective},
                )
                table = result.ranking_table(top=None)
                self.assertEqual(
                    table["candidate_id"].tolist(), ["b", "c", "a"]
                )

    def test_bias_correction_suppresses_unique_category_artifact(self):
        labels = np.repeat([0, 1], 20)
        frame = pd.DataFrame({
            "unique_category": [f"item-{i}" for i in range(40)],
            "real_category": np.repeat(["A", "B"], 20),
        })
        fitted = SimpleNamespace(labels_=labels)
        solution = Solution(
            "example", fitted, frame,
            Schema(categorical=("unique_category", "real_category")),
        )
        result = solution.feature_importance_table().set_index("feature")
        self.assertAlmostEqual(
            result.loc["unique_category", "differentiation"], 0.0
        )
        self.assertGreater(
            result.loc["real_category", "differentiation"], 0.8
        )
        self.assertEqual(
            result.loc["real_category", "measure"],
            "cramers_v_squared_bias_corrected",
        )


if __name__ == "__main__":
    unittest.main()
