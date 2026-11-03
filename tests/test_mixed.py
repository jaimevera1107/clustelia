"""Tests of the M3 milestone: ordinal/binary semantics, weights, prototypes."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from clustelia import (
    AutoCluster,
    CandidateSpec,
    CapabilityError,
    GeometrySpec,
    NoRecommendationError,
    Schema,
    SelectionPolicy,
    load,
    save,
)
from clustelia.geometry import GowerMetric, ReferenceGeometry
from clustelia.planning import DEFAULT_GOWER_FAMILIES
from clustelia.prototypes import GowerPrototypes

LEVELS = ("low", "mid", "high")


def _fast_policy(**overrides):
    config = {"subsamples": 3, "seed_repeats": 2, "shortlist_size": 2}
    config.update(overrides)
    return SelectionPolicy(**config)


def _mixed_groups(n_per_group=60, seed=0):
    """Three well-separated groups over all four variable kinds."""
    rng = np.random.default_rng(seed)
    frames = []
    for group, (centre, colour, level, flag) in enumerate(
        ((0.0, "red", "low", 0), (10.0, "green", "mid", 1),
         (20.0, "blue", "high", 1))
    ):
        frames.append(pd.DataFrame({
            "x": rng.normal(centre, 0.5, n_per_group),
            "colour": colour,
            "level": level,
            "flag": flag,
            "truth": group,
        }))
    frame = pd.concat(frames, ignore_index=True)
    return frame.drop(columns="truth"), frame["truth"].to_numpy()


MIXED_SCHEMA = Schema(
    numeric=["x"], categorical=["colour"], ordinal={"level": LEVELS},
    binary=["flag"],
)


class SchemaDeclarations(unittest.TestCase):
    def test_ordinal_mapping_is_normalised_and_hashable(self):
        schema = Schema(ordinal={"level": ["low", "mid", "high"]})
        self.assertEqual(schema.ordinal, (("level", LEVELS),))
        self.assertEqual(schema.ordinal_levels, {"level": LEVELS})
        hash(schema)

    def test_feature_order_and_nonnumeric(self):
        self.assertEqual(MIXED_SCHEMA.features,
                         ("x", "colour", "level", "flag"))
        self.assertEqual(MIXED_SCHEMA.nonnumeric, ("colour", "level", "flag"))

    def test_invalid_ordinal_declarations(self):
        for ordinal in ({"level": "lmh"}, {"level": ["only"]},
                        {"level": ["a", "a"]}, "level"):
            with self.subTest(ordinal=ordinal), self.assertRaises(ValueError):
                Schema(ordinal=ordinal)

    def test_columns_are_disjoint_across_roles(self):
        with self.assertRaises(ValueError):
            Schema(numeric=["a"], binary=["a"])
        with self.assertRaises(ValueError):
            Schema(categorical=["a"], ordinal={"a": LEVELS})
        with self.assertRaises(ValueError):
            Schema(binary="flag")

    def test_values_outside_declared_semantics_are_rejected(self):
        frame, _ = _mixed_groups(5)
        bad_level = frame.copy()
        bad_level.loc[0, "level"] = "extreme"
        bad_flag = frame.copy()
        bad_flag.loc[0, "flag"] = 2
        for data in (bad_level, bad_flag):
            with self.assertRaises(ValueError):
                AutoCluster(schema=MIXED_SCHEMA, k_range=(2,)).plan(data)

    def test_boolean_binary_is_accepted(self):
        frame, _ = _mixed_groups(5)
        frame["flag"] = frame["flag"].astype(bool)
        plan = AutoCluster(schema=MIXED_SCHEMA, k_range=(2,)).plan(frame)
        self.assertEqual(plan["geometry"]["kind"], "gower")


class WeightedGower(unittest.TestCase):
    def test_hand_computed_distance_with_all_kinds(self):
        frame = pd.DataFrame({
            "x": [0.0, 10.0, 5.0],
            "colour": ["a", "b", "a"],
            "level": ["low", "high", "mid"],
            "flag": [0, 1, 0],
        })
        spec = GeometrySpec("gower", "none", weights={"x": 2.0})
        geometry = ReferenceGeometry(spec, MIXED_SCHEMA).fit(frame)
        D = geometry.distances(geometry.transform(frame))
        # Rows 0-1: x 2*1, colour 1, level 1, flag 1 over weights 2+1+1+1.
        self.assertAlmostEqual(D[0, 1], 5.0 / 5.0)
        # Rows 0-2: x 2*0.5, colour 0, level 0.5; joint absence of flag
        # is excluded, so the denominator is 2+1+1.
        self.assertAlmostEqual(D[0, 2], 1.5 / 4.0)
        np.testing.assert_allclose(D, D.T)
        np.testing.assert_allclose(np.diag(D), 0.0)

    def test_legacy_joint_absences_only_give_zero(self):
        metric = GowerMetric([], [], binary=["a", "b"], incomparable="zero")
        coded = pd.DataFrame({"a": [0, 0, 1], "b": [0, 0, 0]})
        D = metric.fit(coded).distances(coded)
        self.assertEqual(D[0, 1], 0.0)
        self.assertEqual(D[0, 2], 1.0)

    def test_equal_weights_match_unweighted(self):
        frame, _ = _mixed_groups(10)
        plain = ReferenceGeometry(GeometrySpec("gower", "none"),
                                  MIXED_SCHEMA).fit(frame)
        ones = ReferenceGeometry(
            GeometrySpec("gower", "none", weights={"x": 3.0, "colour": 3.0,
                                                   "level": 3.0, "flag": 3.0}),
            MIXED_SCHEMA,
        ).fit(frame)
        np.testing.assert_allclose(
            plain.distances(plain.transform(frame)),
            ones.distances(ones.transform(frame)),
        )

    def test_weight_validation(self):
        with self.assertRaises(ValueError):
            GeometrySpec("euclidean", weights={"x": 1.0})
        with self.assertRaises(ValueError):
            GeometrySpec("gower", "none", weights={"x": 0.0})
        with self.assertRaises(ValueError):
            GeometrySpec("gower", "none", weights=(("x", 1.0), ("x", 2.0)))
        frame, _ = _mixed_groups(5)
        model = AutoCluster(
            schema=MIXED_SCHEMA, k_range=(2,),
            geometry=GeometrySpec("gower", "none", weights={"nope": 1.0}),
        )
        with self.assertRaises(ValueError):
            model.plan(frame)

    def test_euclidean_rejects_nonnumeric_semantics(self):
        frame, _ = _mixed_groups(5)
        with self.assertRaises(CapabilityError):
            AutoCluster(schema=MIXED_SCHEMA, geometry=GeometrySpec(),
                        k_range=(2,)).plan(frame)

    def test_ordinal_transform_rejects_undeclared_levels(self):
        metric = GowerMetric([], [], ordinal={"level": LEVELS})
        with self.assertRaises(ValueError):
            metric.fit(pd.DataFrame({"level": ["low"]})).transform(
                pd.DataFrame({"level": ["extreme"]})
            )


class Prototypes(unittest.TestCase):
    def setUp(self):
        self.frame, self.truth = _mixed_groups()
        self.geometry = ReferenceGeometry(
            GeometrySpec("gower", "none"), MIXED_SCHEMA
        ).fit(self.frame)
        self.coded = self.geometry.transform(self.frame)

    def test_fixed_point_and_determinism(self):
        model = GowerPrototypes(3, 7).fit(self.coded, self.geometry.metric_)
        again = GowerPrototypes(3, 7).fit(self.coded, self.geometry.metric_)
        np.testing.assert_array_equal(model.labels_, again.labels_)
        self.assertTrue(model.converged_)
        np.testing.assert_array_equal(model.predict(self.coded),
                                      model.labels_)
        for cluster in range(3):
            members = self.coded[model.labels_ == cluster]
            prototype = model.prototypes_.iloc[cluster]
            self.assertAlmostEqual(prototype["x"], members["x"].median())
            self.assertEqual(prototype["colour"], members["colour"].mode()[0])
        self.assertEqual(adjusted_rand_score(self.truth, model.labels_), 1.0)

    def test_nominal_only_is_k_modes(self):
        rng = np.random.default_rng(3)
        data = pd.DataFrame({
            c: rng.choice(["a", "b", "c"], 80) for c in ("p", "q", "r")
        })
        schema = Schema(categorical=["p", "q", "r"])
        geometry = ReferenceGeometry(GeometrySpec("gower", "none"),
                                     schema).fit(data)
        coded = geometry.transform(data)
        model = GowerPrototypes(3, 1).fit(coded, geometry.metric_)
        mismatches = (
            coded.to_numpy()[:, None, :]
            != model.prototypes_.to_numpy()[None, :, :]
        ).mean(axis=2)
        np.testing.assert_allclose(
            mismatches[np.arange(80), model.labels_].sum(), model.inertia_
        )

    def test_more_restarts_never_increase_cost(self):
        metric = self.geometry.metric_
        one = GowerPrototypes(4, 11, n_init=1).fit(self.coded, metric)
        many = GowerPrototypes(4, 11, n_init=6).fit(self.coded, metric)
        self.assertLessEqual(many.inertia_, one.inertia_ + 1e-12)

    def test_too_few_distinct_rows(self):
        coded = self.coded.iloc[[0, 0, 0]]
        with self.assertRaises(ValueError):
            GowerPrototypes(2, 0).fit(coded, self.geometry.metric_)


class SearchIntegration(unittest.TestCase):
    def setUp(self):
        self.frame, self.truth = _mixed_groups()

    def test_default_gower_grid_includes_prototypes(self):
        plan = AutoCluster(schema=MIXED_SCHEMA, k_range=(2, 3)).plan(
            self.frame
        )
        algorithms = [c["algorithm"] for c in plan["candidates"]]
        self.assertEqual(DEFAULT_GOWER_FAMILIES, ("medoids", "kprototypes"))
        self.assertEqual(algorithms, ["medoids", "kprototypes"] * 2)

    def test_prototypes_excluded_from_euclidean(self):
        X = self.frame[["x"]].assign(y=np.arange(len(self.frame)) % 7)
        plan = AutoCluster(
            candidates=(CandidateSpec("kmeans", 2),
                        CandidateSpec("kprototypes", 2)),
        ).plan(X)
        self.assertEqual(plan["exclusions"][0]["reason"],
                         "incompatible geometry")

    def test_end_to_end_recovery_assignment_and_profiles(self):
        model = AutoCluster(
            schema=MIXED_SCHEMA, k_range=(2, 3, 4), random_state=5,
            selection=_fast_policy(),
        ).fit(self.frame)
        result = model.result_
        self.assertEqual(result.selection_status, "recommended")
        self.assertIn("kprototypes", result.fitted_algorithms)
        self.assertEqual(adjusted_rand_score(self.truth, model.labels_), 1.0)
        prototypes_id = next(
            i for i in result.candidate_ids
            if result.select(i).pipeline_.spec.algorithm == "kprototypes"
        )
        solution = result.select(prototypes_id)
        np.testing.assert_array_equal(solution.assign(self.frame),
                                      solution.labels_)
        unseen = self.frame.iloc[:1].copy()
        unseen.loc[unseen.index[0], "colour"] = "violet"
        with self.assertRaises(ValueError):
            solution.assign(unseen)
        profile = solution.describe()[0]
        self.assertIn(profile["ordinal"]["level"]["median"], LEVELS)
        self.assertEqual(list(profile["ordinal"]["level"]["proportions"]),
                         list(LEVELS))
        self.assertIn(profile["binary"]["flag"], (0.0, 1.0))

    def test_persistence_round_trip(self):
        model = AutoCluster(
            schema=MIXED_SCHEMA,
            candidates=(CandidateSpec("kprototypes", 3),),
            selection=_fast_policy(), random_state=2,
        ).fit(self.frame)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pkl"
            save(model, path)
            restored = load(path, trusted=True)
        np.testing.assert_array_equal(restored.predict(self.frame),
                                      model.labels_)

    def test_predict_follows_the_recommendation(self):
        model = AutoCluster(
            schema=MIXED_SCHEMA,
            candidates=(CandidateSpec("kprototypes", 3),),
            selection=_fast_policy(), random_state=2,
        ).fit(self.frame)
        np.testing.assert_array_equal(model.predict(self.frame),
                                      model.labels_)
        model.result_.recommendation = None
        with self.assertRaises(NoRecommendationError):
            model.predict(self.frame)

    def test_medoids_accept_ordinal_and_binary(self):
        model = AutoCluster(
            schema=MIXED_SCHEMA, candidates=(CandidateSpec("medoids", 3),),
            selection=_fast_policy(), random_state=3,
        ).fit(self.frame)
        solution = model.result_.select(model.result_.candidate_ids[0])
        np.testing.assert_array_equal(solution.assign(self.frame),
                                      solution.labels_)


if __name__ == "__main__":
    unittest.main()
