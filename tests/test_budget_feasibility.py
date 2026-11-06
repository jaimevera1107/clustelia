"""Tests of the fit-budget feasibility check announced before fitting."""

import unittest
import warnings

from sklearn.datasets import make_blobs

from clustelia import AutoCluster, SearchBudget, SelectionPolicy
from clustelia.search import budget_shortfall_message, fit_requirements


class FitRequirements(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(120, centers=3, cluster_std=0.6, random_state=0)

    def test_plan_reports_requirements_consistent_with_reserves(self):
        plan = AutoCluster(budget=SearchBudget(max_fits=320)).plan(self.X)
        req = plan["fit_requirements"]
        n_candidates = len(plan["candidates"])
        self.assertEqual(req["exploration"], n_candidates)
        self.assertEqual(req["structure"], plan["reserved_structure_fits"])
        self.assertEqual(req["confirmation"],
                         plan["maximum_confirmation_fits"])
        self.assertEqual(req["minimum"], req["exploration"] + req["structure"])
        self.assertEqual(req["maximum"], req["minimum"] + req["confirmation"])
        self.assertEqual(req["max_fits"], 320)
        self.assertTrue(req["sufficient"])

    def test_holdout_doubles_exploration(self):
        model = AutoCluster(objective="probabilistic", families=("gmm",),
                            probabilistic_metric="heldout_loglikelihood",
                            k_range=(2, 3))
        req = model.plan(self.X)["fit_requirements"]
        self.assertEqual(req["exploration"], 2 * 2)
        self.assertEqual(req["structure"], 0)

    def test_insufficient_budget_is_detected_before_fitting(self):
        model = AutoCluster(
            selection=SelectionPolicy(null_replicates=19),
            budget=SearchBudget(max_fits=320),
        )
        req = model.plan(self.X)["fit_requirements"]
        self.assertFalse(req["sufficient"])
        self.assertGreater(req["minimum"], 320)
        message = budget_shortfall_message(req)
        self.assertIn("max_fits=320", message)
        self.assertIn(str(req["minimum"]), message)
        self.assertIn(str(req["maximum"]), message)

    def test_fit_warns_and_names_the_shortfall(self):
        model = AutoCluster(
            selection=SelectionPolicy(null_replicates=19),
            budget=SearchBudget(max_fits=320),
            random_state=0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(self.X)
        texts = [str(w.message) for w in caught
                 if issubclass(w.category, UserWarning)]
        self.assertTrue(any("cannot complete" in t for t in texts), texts)
        result = model.result_
        self.assertEqual(result.selection_status, "incomplete")
        reasons = result.metadata["incomplete_reasons"]
        self.assertTrue(any("max_fits=320 is below" in r for r in reasons))
        # It stops early instead of spending the whole budget.
        self.assertLess(result.metadata["fits_used"], 320)

    def test_sufficient_budget_does_not_warn(self):
        model = AutoCluster(
            families=("kmeans",), k_range=(2, 3, 4),
            selection=SelectionPolicy(null_replicates=3, subsamples=3,
                                      seed_repeats=1),
            random_state=0,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(self.X)
        self.assertFalse(any("cannot complete" in str(w.message)
                             for w in caught))
        self.assertNotIn("incomplete", model.result_.selection_status)

    def test_requirements_helper_matches_plan(self):
        model = AutoCluster(budget=SearchBudget(max_fits=50))
        plan = model._prepare(self.X)
        req = fit_requirements(plan, "compact", False)
        self.assertEqual(req, model.plan(self.X)["fit_requirements"])
