"""End-to-end guarantees an auditor should be able to check quickly.

Component-level determinism is covered elsewhere; these tests assert the
properties a user relies on from the outside: two identical runs agree on
everything, a different seed is genuinely a different draw, the demo in
``docs/ABSTENTION_DEMO.md`` still reaches the decisions it documents, and
the estimator works for categorical-only tables and inside a
scikit-learn ``Pipeline``.
"""

import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_blobs
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clustelia import (
    AutoCluster,
    GeometrySpec,
    Schema,
    SelectionPolicy,
    cluster,
)

FAST = SelectionPolicy(null_replicates=3, subsamples=3, seed_repeats=2)


def _blobs(seed: int = 0):
    return make_blobs(200, n_features=3, centers=3, cluster_std=0.6,
                      random_state=seed)[0]


class EndToEndReproducibility(unittest.TestCase):
    """Same inputs and seed must give byte-identical results."""

    def setUp(self):
        self.X = _blobs()

    def _fit(self, seed: int):
        return AutoCluster(
            families=("kmeans", "agglomerative"), k_range=(2, 3, 4),
            selection=FAST, random_state=seed,
        ).fit(self.X)

    @staticmethod
    def _comparable(table: pd.DataFrame) -> pd.DataFrame:
        """Drop the only column that cannot be reproducible.

        ``fit_seconds`` is measured wall-clock time; everything else in
        the table is a function of the data, the configuration and the
        seed, and must match exactly.
        """
        return table.drop(columns=["fit_seconds"])

    def test_two_runs_with_the_same_seed_agree_on_everything(self):
        first, second = self._fit(0), self._fit(0)
        self.assertEqual(first.result_.selection_status,
                         second.result_.selection_status)
        np.testing.assert_array_equal(first.labels_, second.labels_)
        pd.testing.assert_frame_equal(
            self._comparable(first.result_.candidates_table()),
            self._comparable(second.result_.candidates_table()),
            check_exact=True,
        )
        for key in ("selected_k", "fits_used", "null_family",
                    "selection_warnings", "incomplete_reasons"):
            self.assertEqual(first.result_.metadata[key],
                             second.result_.metadata[key], key)
        self.assertEqual(first.result_.metadata["structure_test"],
                         second.result_.metadata["structure_test"])
        self.assertEqual(first.result_.recommendation.candidate_id,
                         second.result_.recommendation.candidate_id)

    def test_prediction_is_reproducible(self):
        # k-means only: transductive families have no out-of-sample
        # assignment, which is a capability limit, not a seeding issue.
        def fit():
            return AutoCluster(families=("kmeans",), k_range=(2, 3, 4),
                               selection=FAST, random_state=0).fit(self.X)

        new = _blobs(seed=11)
        np.testing.assert_array_equal(fit().predict(new), fit().predict(new))

    def test_a_different_seed_is_a_different_draw(self):
        # Reproducibility must not come from the seed being ignored: the
        # stochastic parts (null references, subsamples) have to move.
        one = self._fit(0).result_.metadata["structure_test"]
        two = self._fit(7).result_.metadata["structure_test"]
        self.assertNotEqual(one["null_maxima"], two["null_maxima"])

    def test_the_convenience_api_is_reproducible_too(self):
        first = cluster(self.X, effort="quick", random_state=3)
        second = cluster(self.X, effort="quick", random_state=3)
        pd.testing.assert_frame_equal(
            self._comparable(first.candidates_table()),
            self._comparable(second.candidates_table()),
            check_exact=True,
        )


class DocumentedDecisions(unittest.TestCase):
    """The demonstration in docs/ABSTENTION_DEMO.md must stay true."""

    def test_noise_abstains_and_blobs_are_recommended(self):
        noise = np.random.default_rng(0).normal(size=(300, 4))
        self.assertNotEqual(
            cluster(noise, random_state=0).selection_status, "recommended"
        )
        blobs = make_blobs(300, n_features=4, centers=3, cluster_std=0.6,
                           random_state=0)[0]
        self.assertEqual(
            cluster(blobs, random_state=0).selection_status, "recommended"
        )

    def test_the_copula_reference_rejects_a_skewed_single_mode(self):
        skewed = np.exp(np.random.default_rng(0).normal(size=(300, 3)))
        result = cluster(
            skewed, random_state=0,
            selection=SelectionPolicy(null_family="gaussian_copula"),
        )
        self.assertNotEqual(result.selection_status, "recommended")
        self.assertFalse(result.metadata["structure_test"]["passed"])


class SupportedInputs(unittest.TestCase):
    """Cases the documentation claims work, asserted rather than assumed."""

    def test_categorical_only_tables_are_supported(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({
            "a": rng.choice(list("xyz"), 150),
            "b": rng.choice(list("pq"), 150),
            "c": rng.choice(list("mnop"), 150),
        })
        model = AutoCluster(
            schema=Schema(categorical=["a", "b", "c"]),
            geometry=GeometrySpec(kind="gower", scaling="none"),
            k_range=(2, 3), selection=FAST, random_state=0,
        ).fit(frame)
        # No numeric column is required; the search must reach a verdict.
        self.assertIn(model.result_.selection_status,
                      {"recommended", "ambiguous", "no_admissible_solution"})
        self.assertGreaterEqual(len(model.result_.candidates_table()), 2)
        self.assertEqual(model.result_.metadata["geometry"]["kind"], "gower")

    def test_it_works_inside_a_scikit_learn_pipeline(self):
        X = load_iris(as_frame=True).data
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("cluster", AutoCluster(families=("kmeans",), k_range=(2, 3, 4),
                                    selection=FAST, random_state=0)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X)
        labels = pipeline.predict(X)
        self.assertEqual(len(labels), len(X))
        self.assertEqual(pipeline["cluster"].result_.selection_status,
                         "recommended")
        # get_params/set_params must survive cloning by the Pipeline.
        self.assertEqual(pipeline.get_params()["cluster__random_state"], 0)


if __name__ == "__main__":
    unittest.main()
