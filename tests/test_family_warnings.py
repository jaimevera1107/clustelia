"""Warn when a whole algorithm family fails during exploration."""

import unittest

from sklearn.datasets import make_blobs

from clustelia import AutoCluster, SearchBudget, SelectionPolicy


def _fit(max_pairwise_mb):
    X, _ = make_blobs(300, centers=3, cluster_std=0.5, random_state=2)
    budget = SearchBudget(max_pairwise_mb=max_pairwise_mb,
                          metric_sample_size=50)
    return AutoCluster(
        k_range=(2, 3, 4, 5), random_state=1, budget=budget,
        selection=SelectionPolicy(subsamples=3, seed_repeats=2),
    ).fit(X).result_


class FamilyWarnings(unittest.TestCase):
    def test_memory_guard_drop_is_reported(self):
        result = _fit(1.0)
        table = result.candidates_table()
        failed = set(table.loc[table["status"] == "failed", "algorithm"])
        self.assertEqual(failed, {"agglomerative", "medoids"})
        warnings = result.metadata["selection_warnings"]
        for family in failed:
            matching = [w for w in warnings if f"{family} candidates" in w]
            self.assertEqual(len(matching), 1, family)
            self.assertIn("continued without that family", matching[0])
            self.assertIn("max_pairwise_mb", matching[0])
        self.assertEqual(result.selection_status, "recommended")

    def test_no_warning_when_every_family_runs(self):
        result = _fit(256)
        self.assertEqual(result.metadata["selection_warnings"], [])

    def test_stage_timing_metadata(self):
        result = _fit(256)
        stages = result.metadata["stage_seconds"]
        self.assertEqual(set(stages), {"exploration", "validation",
                                       "composition", "randomness",
                                       "structure"})
        self.assertGreater(stages["structure"], 0)
        self.assertGreaterEqual(result.metadata["structure_test_seconds"],
                                stages["structure"] * 0.5)


if __name__ == "__main__":
    unittest.main()
