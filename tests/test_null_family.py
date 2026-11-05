"""Tests of the ``null_family`` and ``boundary_k_policy`` options.

Both options come from the 2026-11 validation study (``docs/VALIDATION.md``):
the Gaussian-copula reference preserves every marginal, and the
boundary-k abstention keeps an unresolved upper bound of ``k`` from being
reported as a validated recommendation.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from clustelia import AutoCluster, GeometrySpec, Schema, SelectionPolicy
from clustelia.structure import (
    NULL_FAMILIES,
    generate_null,
    null_family,
)


def _fast_policy(**overrides):
    config = {"subsamples": 3, "seed_repeats": 2, "shortlist_size": 3}
    config.update(overrides)
    return SelectionPolicy(**config)


def _skewed_frame(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n_rows)
    return pd.DataFrame({
        "a": np.exp(base),
        "b": np.exp(0.8 * base + 0.6 * rng.standard_normal(n_rows)),
        "c": rng.standard_normal(n_rows),
        "const": 1.0,
    })


class NullFamilyResolution(unittest.TestCase):
    def test_auto_follows_the_geometry_default(self):
        for kind, family in NULL_FAMILIES.items():
            self.assertEqual(null_family(kind), family)
            self.assertEqual(null_family(kind, "auto"), family)

    def test_explicit_family_is_returned_for_euclidean(self):
        self.assertEqual(null_family("euclidean", "gaussian_copula"),
                         "gaussian_copula")
        self.assertEqual(null_family("euclidean", "permutation"),
                         "permutation")

    def test_numeric_families_are_rejected_for_gower(self):
        self.assertEqual(null_family("gower", "permutation"), "permutation")
        with self.assertRaises(ValueError):
            null_family("gower", "gaussian_copula")
        with self.assertRaises(ValueError):
            null_family("gower", "pca_uniform")

    def test_policy_validation(self):
        self.assertEqual(SelectionPolicy().null_family, "auto")
        self.assertEqual(SelectionPolicy().boundary_k_policy, "warn")
        with self.assertRaises(ValueError):
            SelectionPolicy(null_family="bootstrap")
        with self.assertRaises(ValueError):
            SelectionPolicy(boundary_k_policy="ignore")


class GaussianCopulaNull(unittest.TestCase):
    def setUp(self):
        self.frame = _skewed_frame()
        self.schema = Schema(numeric=["a", "b", "c", "const"])

    def _null(self, seed: int = 1) -> pd.DataFrame:
        return generate_null(
            "gaussian_copula", self.frame, self.schema,
            np.random.default_rng(seed),
        )

    def test_marginals_are_preserved_exactly(self):
        null = self._null()
        self.assertEqual(list(null.columns), list(self.frame.columns))
        self.assertTrue(null.index.equals(self.frame.index))
        for column in ("a", "b", "c"):
            np.testing.assert_array_equal(
                np.sort(null[column].to_numpy()),
                np.sort(self.frame[column].to_numpy()),
            )
            # It is a genuine resample, not a copy.
            self.assertFalse(
                np.array_equal(null[column].to_numpy(),
                               self.frame[column].to_numpy())
            )
        np.testing.assert_array_equal(null["const"], self.frame["const"])

    def test_rank_correlation_is_reproduced(self):
        observed = self.frame[["a", "b", "c"]].corr(method="spearman")
        drawn = pd.concat(
            [self._null(seed)[["a", "b", "c"]] for seed in range(5)]
        ).corr(method="spearman")
        self.assertAlmostEqual(observed.loc["a", "b"], drawn.loc["a", "b"],
                               delta=0.08)
        self.assertGreater(drawn.loc["a", "b"], 0.6)
        self.assertLess(abs(drawn.loc["a", "c"]), 0.15)

    def test_is_reproducible_from_the_seed(self):
        pd.testing.assert_frame_equal(self._null(7), self._null(7))

    def test_non_numeric_features_are_rejected(self):
        frame = self.frame.assign(colour=["r", "g"] * 100)
        schema = Schema(numeric=["a", "b", "c"], categorical=["colour"])
        with self.assertRaises(ValueError):
            generate_null("gaussian_copula", frame, schema,
                          np.random.default_rng(0))

    def test_unknown_family_names_the_options(self):
        with self.assertRaises(ValueError) as caught:
            generate_null("bootstrap", self.frame, self.schema,
                          np.random.default_rng(0))
        self.assertIn("gaussian_copula", str(caught.exception))


class CopulaSearch(unittest.TestCase):
    def test_separated_blobs_are_still_recommended(self):
        X, _ = make_blobs(240, centers=3, cluster_std=0.5, random_state=3)
        result = AutoCluster(
            random_state=3,
            selection=_fast_policy(null_family="gaussian_copula"),
        ).fit(X).result_
        self.assertEqual(result.selection_status, "recommended")
        self.assertEqual(result.metadata["null_family"], "gaussian_copula")
        self.assertEqual(result.metadata["structure_test"]["family"],
                         "gaussian_copula")
        self.assertEqual(result.summary()["null_family"], "gaussian_copula")

    def test_gower_geometry_rejects_numeric_families(self):
        X = pd.DataFrame({
            "x": np.random.default_rng(0).standard_normal(60),
            "colour": ["r", "g", "b"] * 20,
        })
        search = AutoCluster(
            geometry=GeometrySpec(kind="gower", scaling="none"),
            schema=Schema(numeric=["x"], categorical=["colour"]),
            selection=_fast_policy(null_family="gaussian_copula"),
            random_state=0,
        )
        with self.assertRaises(ValueError):
            search.fit(X)


class BoundaryKPolicy(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(300, centers=6, cluster_std=0.4,
                               random_state=1)

    def test_warn_keeps_the_previous_behaviour(self):
        result = AutoCluster(
            k_range=(2, 3, 4), random_state=0, selection=_fast_policy(),
        ).fit(self.X).result_
        self.assertEqual(result.metadata["selected_k"], 4)
        self.assertTrue(result.metadata["selected_k_at_boundary"])
        self.assertEqual(result.metadata["boundary_k_policy"], "warn")
        self.assertIn("widen k_range",
                      result.metadata["selection_warnings"][0])
        self.assertEqual(result.metadata["incomplete_reasons"], [])

    def test_abstain_withholds_confirmation_and_is_incomplete(self):
        result = AutoCluster(
            k_range=(2, 3, 4), random_state=0,
            selection=_fast_policy(boundary_k_policy="abstain"),
        ).fit(self.X).result_
        self.assertEqual(result.selection_status, "incomplete")
        self.assertIsNone(result.recommendation)
        # Diagnostics survive the abstention.
        self.assertEqual(result.metadata["selected_k"], 4)
        self.assertTrue(result.metadata["selected_k_at_boundary"])
        self.assertEqual([e["k"] for e in result.metadata["gap_table"]],
                         [2, 3, 4])
        self.assertTrue(any(
            "boundary_k_policy='abstain'" in reason
            for reason in result.metadata["incomplete_reasons"]
        ))
        # No confirmation fit is spent on an unresolved upper bound.
        breakdown = result.metadata["fit_breakdown"]
        self.assertEqual(breakdown["composition"], 0)
        self.assertEqual(breakdown["randomness"], 0)
        self.assertEqual(result.metadata["confirmation_candidates"], [])
        self.assertTrue(result.summary()["selected_k_at_boundary"])

    def test_abstain_is_inert_inside_the_range(self):
        result = AutoCluster(
            k_range=range(2, 9), random_state=0,
            selection=_fast_policy(boundary_k_policy="abstain"),
        ).fit(self.X).result_
        self.assertEqual(result.selection_status, "recommended")
        self.assertFalse(result.metadata["selected_k_at_boundary"])
        self.assertEqual(result.metadata["selection_warnings"], [])
