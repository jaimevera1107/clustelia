"""Tests of the gap-statistic choice of the number of clusters (P2)."""

import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustelia import AutoCluster, CandidateSpec, SelectionPolicy
from clustelia.evaluation import within_dispersion
from clustelia.gap import GapEntry, gap_table, select_k


def _fast_policy(**overrides):
    config = {"subsamples": 3, "seed_repeats": 2, "shortlist_size": 3}
    config.update(overrides)
    return SelectionPolicy(**config)


def _row(identifier, k, dispersion):
    return {"candidate_id": identifier, "status": "fitted",
            "parameters": {"k": k}, "within_dispersion": dispersion}


class GapStatistic(unittest.TestCase):
    def test_within_dispersion_by_hand(self):
        # Two clusters {0, 1} at distance 2 and {2} alone.
        distances = np.array([[0.0, 2.0, 9.0], [2.0, 0.0, 9.0],
                              [9.0, 9.0, 0.0]])
        self.assertAlmostEqual(
            within_dispersion(distances, np.array([0, 0, 1])), 4.0 / 4.0
        )

    def test_table_keeps_the_best_gap_per_k(self):
        rows = [_row("a", 2, 1.0), _row("b", 2, 0.5), _row("c", 3, 1.0)]
        nulls = {"a": [np.e, np.e], "b": [np.e, np.e], "c": [1.0, 1.0]}
        table = gap_table(rows, nulls)
        self.assertEqual([e.k for e in table], [2, 3])
        self.assertEqual(table[0].candidate_id, "b")
        self.assertAlmostEqual(table[0].gap, 1.0 - np.log(0.5))
        self.assertAlmostEqual(table[0].standard_error, 0.0)

    def test_degenerate_entries_are_skipped(self):
        rows = [_row("a", 2, None), _row("b", 3, 1.0)]
        self.assertEqual(gap_table(rows, {"a": [1.0, 1.0], "b": [None]}), [])

    def test_one_standard_error_rule(self):
        table = [GapEntry(2, "a", 0.5, 0.1), GapEntry(3, "b", 0.9, 0.1),
                 GapEntry(4, "c", 0.95, 0.1)]
        # Gap(2) < Gap(3) - s3, Gap(3) >= Gap(4) - s4: choose 3.
        self.assertEqual(select_k(table), 3)
        self.assertEqual(select_k(table[:1]), 2)
        self.assertIsNone(select_k([]))

    def test_policy_resolution(self):
        self.assertTrue(SelectionPolicy().uses_gap)
        self.assertFalse(SelectionPolicy(null_scope="family").uses_gap)
        self.assertFalse(SelectionPolicy(null_replicates=0).uses_gap)
        self.assertFalse(SelectionPolicy.p0().uses_gap)
        with self.assertRaises(ValueError):
            SelectionPolicy(k_selection="gap", null_replicates=1)
        with self.assertRaises(ValueError):
            SelectionPolicy(k_selection="elbow")


class GapSearch(unittest.TestCase):
    def test_gap_recovers_blobs_that_silhouette_merges(self):
        # Six blobs whose centres pair up: the silhouette band picks k=2.
        X, y = make_blobs(300, centers=6, cluster_std=0.5,
                          random_state=1022)
        gap = AutoCluster(random_state=1022,
                          selection=_fast_policy()).fit(X).result_
        silhouette = AutoCluster(
            random_state=1022,
            selection=_fast_policy(k_selection="silhouette"),
        ).fit(X).result_
        self.assertEqual(gap.metadata["selected_k"], 6)
        chosen = gap.select(gap.recommendation.candidate_id)
        self.assertGreater(adjusted_rand_score(y, chosen.labels_), 0.9)
        merged = silhouette.select(silhouette.recommendation.candidate_id)
        self.assertLess(adjusted_rand_score(y, merged.labels_), 0.5)

    def test_only_the_selected_k_is_confirmed(self):
        X, _ = make_blobs(200, centers=3, cluster_std=0.5, random_state=4)
        result = AutoCluster(random_state=2,
                             selection=_fast_policy()).fit(X).result_
        table = result.candidates_table()
        chosen = result.metadata["selected_k"]
        ks = table["parameters"].map(lambda p: p["k"])
        self.assertEqual(chosen, 3)
        self.assertTrue((ks[table["confirmed"]] == chosen).all())
        others = table[(ks != chosen) & (table["status"] == "fitted")]
        self.assertTrue(all(
            f"gap statistic selects k={chosen}" in reasons
            for reasons in others["rejection_reasons"]
        ))
        self.assertEqual(result.metadata["k_selection"], "gap")
        self.assertEqual([e["k"] for e in result.metadata["gap_table"]],
                         list(range(2, 9)))

    def test_identical_partitions_are_all_confirmed(self):
        # kmeans, Ward and medoids find the same 6 clusters; medoids is
        # unstable. Deduplication used to keep only medoids and abstain.
        X, y = make_blobs(300, n_features=5, centers=6, cluster_std=0.5,
                          random_state=29)
        result = AutoCluster(random_state=29).fit(X).result_
        table = result.candidates_table()
        confirmed = table[table["confirmed"]]
        self.assertEqual(sorted(confirmed["algorithm"]),
                         ["agglomerative", "kmeans", "medoids"])
        self.assertEqual(result.selection_status, "recommended")
        chosen = result.select(result.recommendation.candidate_id)
        self.assertEqual(adjusted_rand_score(y, chosen.labels_), 1.0)

    def test_boundary_k_is_flagged(self):
        X, _ = make_blobs(300, centers=6, cluster_std=0.4, random_state=1)
        result = AutoCluster(k_range=(2, 3, 4), random_state=0,
                             selection=_fast_policy()).fit(X).result_
        self.assertEqual(result.metadata["selected_k"], 4)
        self.assertIn("widen k_range",
                      result.metadata["selection_warnings"][0])
        inside = AutoCluster(k_range=range(2, 9), random_state=0,
                             selection=_fast_policy()).fit(X).result_
        self.assertEqual(inside.metadata["selection_warnings"], [])

    def test_failed_null_test_spends_no_confirmation(self):
        X = np.random.default_rng(0).uniform(size=(200, 2))
        result = AutoCluster(random_state=0,
                             selection=_fast_policy()).fit(X).result_
        self.assertIsNone(result.recommendation)
        breakdown = result.metadata["fit_breakdown"]
        self.assertEqual(breakdown["composition"], 0)
        self.assertEqual(breakdown["randomness"], 0)
        self.assertIsNone(result.metadata["selected_k"])

    def test_explicit_candidates_with_one_k(self):
        X, _ = make_blobs(150, centers=3, cluster_std=0.5, random_state=0)
        result = AutoCluster(
            candidates=(CandidateSpec("kmeans", 3),
                        CandidateSpec("agglomerative", 3)),
            selection=_fast_policy(),
        ).fit(X).result_
        self.assertEqual(result.metadata["selected_k"], 3)
        self.assertEqual(result.selection_status, "recommended")


if __name__ == "__main__":
    unittest.main()


class GapTableEdges(unittest.TestCase):
    def test_skips_unfitted_rows_and_rows_without_null_fits(self):
        def row(identifier, k, status="fitted"):
            return {"candidate_id": identifier, "status": status,
                    "parameters": {"k": k}, "within_dispersion": 1.0}

        rows = [row("a", 2), row("b", 3, status="failed"), row("c", 4)]
        nulls = {"a": [2.0, 2.2, 2.1], "b": [2.0, 2.0, 2.0]}
        table = gap_table(rows, nulls)
        self.assertEqual([e.candidate_id for e in table], ["a"])
