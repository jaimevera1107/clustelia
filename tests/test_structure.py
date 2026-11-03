"""Tests of the null-reference structure test (policy P1)."""

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
from clustelia.structure import (
    StructureOutcome,
    StructureTester,
    family_grid,
    generate_null,
    structure_reserve,
)


def _fast_policy(**overrides):
    config = {"subsamples": 3, "seed_repeats": 2, "shortlist_size": 2}
    config.update(overrides)
    return SelectionPolicy(**config)


def _structure_rows(result):
    table = result.candidates_table()
    return table[table["structure_test"].notna()]


class NullGeneration(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.frame = pd.DataFrame({
            "a": rng.normal(size=50),
            "b": rng.normal(size=50) * 3,
            "const": 1.0,
            "id": np.arange(50),
        })
        self.schema = Schema(numeric=["a", "b", "const"], identifiers=["id"])

    def test_numeric_families_keep_shape_constants_and_identifiers(self):
        for family in ("pca_uniform", "permutation"):
            null = generate_null(family, self.frame, self.schema,
                                 np.random.default_rng(1))
            with self.subTest(family=family):
                self.assertEqual(null.shape, self.frame.shape)
                self.assertTrue((null["const"] == 1.0).all())
                np.testing.assert_array_equal(null["id"], self.frame["id"])
                self.assertFalse(null["a"].equals(self.frame["a"]))

    def test_permutation_keeps_marginals(self):
        null = generate_null("permutation", self.frame, self.schema,
                             np.random.default_rng(2))
        for column in ("a", "b"):
            np.testing.assert_allclose(np.sort(null[column]),
                                       np.sort(self.frame[column]))

    def test_pca_uniform_stays_in_the_principal_box(self):
        null = generate_null("pca_uniform", self.frame, self.schema,
                             np.random.default_rng(3))
        self.assertEqual(len(null), len(self.frame))
        self.assertTrue(np.isfinite(null[["a", "b"]].to_numpy()).all())

    def test_invalid_family_and_nonnumeric_features(self):
        with self.assertRaises(ValueError):
            generate_null("bootstrap", self.frame, self.schema,
                          np.random.default_rng(0))
        mixed = self.frame.assign(c=["x", "y"] * 25)
        schema = Schema(numeric=["a", "b", "const"], categorical=["c"],
                        identifiers=["id"])
        with self.assertRaises(ValueError):
            generate_null("pca_uniform", mixed, schema,
                          np.random.default_rng(0))

    def test_reserve_covers_the_largest_family_grid(self):
        specs = [CandidateSpec("kmeans", k) for k in (2, 3, 4)] + [
            CandidateSpec("medoids", 2),
            CandidateSpec("kmeans", 3, pca_variance=0.9),
        ]
        policy = SelectionPolicy()
        self.assertEqual(family_grid(specs[1], specs), specs[:3])
        self.assertEqual(structure_reserve(specs, policy, "compact"), 45)
        family = SelectionPolicy(null_scope="family")
        self.assertEqual(structure_reserve(specs, family, "compact"), 27)
        with self.assertRaises(ValueError):
            SelectionPolicy(null_scope="shortlist")
        self.assertEqual(structure_reserve(specs, policy, "density"), 0)
        self.assertEqual(
            structure_reserve(specs, SelectionPolicy(null_replicates=0),
                              "compact"),
            0,
        )

    def test_p0_disables_the_test(self):
        policy = SelectionPolicy.p0()
        self.assertEqual(policy.null_replicates, 0)
        self.assertEqual(policy.version, "P0-alpha3")
        with self.assertRaises(ValueError):
            SelectionPolicy(null_replicates=-1)


class SearchBehaviour(unittest.TestCase):
    def test_structureless_data_are_not_recommended(self):
        for name, X in (
            ("uniform", np.random.default_rng(0).uniform(size=(200, 2))),
            ("gaussian", np.random.default_rng(1).normal(size=(200, 2))),
        ):
            with self.subTest(data=name):
                result = AutoCluster(
                    random_state=0, selection=_fast_policy()
                ).fit(X).result_
                self.assertIsNone(result.recommendation)
                self.assertNotEqual(result.selection_status, "recommended")
                for record in _structure_rows(result)["structure_test"]:
                    self.assertFalse(record["passed"])
                    self.assertIn("null reference", record["reason"])

    def test_separated_clusters_pass_and_are_deterministic(self):
        X, _ = make_blobs(200, centers=3, cluster_std=0.5, random_state=4)
        model = AutoCluster(random_state=2, selection=_fast_policy())
        first = model.fit(X).result_
        self.assertEqual(first.selection_status, "recommended")
        record = first.metadata["structure_test"]
        self.assertTrue(record["passed"])
        self.assertEqual(record["family"], "pca_uniform")
        self.assertEqual(record["replicates"], 9)
        self.assertEqual(len(record["grid"]), 21)
        self.assertTrue(all(record["observed"] > v
                            for v in record["null_maxima"]))
        self.assertGreater(first.metadata["fit_breakdown"]["structure"], 0)
        second = AutoCluster(random_state=2,
                             selection=_fast_policy()).fit(X).result_
        self.assertEqual(
            _structure_rows(first)["structure_test"].tolist(),
            _structure_rows(second)["structure_test"].tolist(),
        )

    def test_gower_uses_the_permutation_reference(self):
        rng = np.random.default_rng(0)
        groups = np.repeat([0, 1, 2], 50)
        frame = pd.DataFrame({
            "x": rng.normal(groups * 10.0, 0.5),
            "c": np.array(["a", "b", "c"])[groups],
        })
        result = AutoCluster(k_range=(2, 3, 4), random_state=1,
                             selection=_fast_policy()).fit(frame).result_
        self.assertEqual(result.selection_status, "recommended")
        self.assertEqual(result.metadata["structure_test"]["family"],
                         "permutation")
        noise = pd.DataFrame({
            "x": rng.normal(size=150),
            "c": rng.choice(["a", "b", "c"], 150),
        })
        result = AutoCluster(k_range=(2, 3, 4), random_state=1,
                             selection=_fast_policy()).fit(noise).result_
        self.assertIsNone(result.recommendation)

    def test_other_objectives_skip_the_test(self):
        X, _ = make_blobs(150, centers=3, cluster_std=0.5, random_state=0)
        result = AutoCluster(objective="probabilistic", k_range=(2, 3),
                             selection=_fast_policy()).fit(X).result_
        self.assertTrue(result.candidates_table()["structure_test"]
                        .isna().all())
        self.assertEqual(result.metadata["fit_breakdown"]["structure"], 0)

    def test_exhausted_test_makes_the_result_incomplete(self):
        X, _ = make_blobs(150, centers=3, cluster_std=0.5, random_state=0)
        exhausted = StructureOutcome(
            observed=0.5, family="pca_uniform", grid=["x"], exhausted=True,
            reason="null reference test not completed",
        )
        with patch.object(StructureTester, "test", return_value=exhausted):
            result = AutoCluster(selection=_fast_policy()).fit(X).result_
        self.assertEqual(result.selection_status, "incomplete")
        self.assertIsNone(result.recommendation)

    def test_tester_stops_when_the_budget_is_gone(self):
        class Spent:
            def available(self):
                return False

        X, _ = make_blobs(60, centers=2, random_state=0)
        frame = pd.DataFrame(X)
        tester = StructureTester(
            Spent(), frame, Schema(numeric=[0, 1]), GeometrySpec(),
            np.arange(60), SelectionPolicy(), 256, "record",
        )
        outcome = tester.test(0.9, [CandidateSpec("kmeans", 2)])
        self.assertTrue(outcome.exhausted)
        self.assertFalse(outcome.passed)

    def test_plan_reports_the_reserve(self):
        X, _ = make_blobs(60, centers=2, random_state=0)
        plan = AutoCluster(budget=SearchBudget(max_fits=200)).plan(X)
        # 21 default candidates refitted in each of nine replicates.
        self.assertEqual(plan["reserved_structure_fits"], 189)


if __name__ == "__main__":
    unittest.main()
