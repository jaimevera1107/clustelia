"""Public estimator and the orchestration of one search run.

:class:`AutoCluster` is a thin scikit-learn compatible facade. The work of
a single ``fit`` call is delegated to :class:`_SearchRun`, which composes
the planner, the fit executor, an objective scorer, the stability
confirmer and the selection policy.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import asdict
from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted

from ._environment import dependency_versions
from ._seeding import candidate_id, stable_seed
from ._validation import is_integer
from ._version import __version__
from .adapters import FittedCandidate
from .config import (
    CandidateSpec,
    GeometrySpec,
    Objective,
    Schema,
    SearchBudget,
    SelectionPolicy,
    candidate_parameters,
)
from .confirmation import StabilityConfirmer
from .evaluation import partition_metrics
from .exceptions import (
    CapabilityError,
    NoRecommendationError,
    SearchFailedError,
)
from .execution import BACKEND_ERRORS, FitExecutor
from .gap import gap_table, select_k
from .geometry import ReferenceGeometry, guard_pairwise
from .planning import CandidatePlanner, SearchPlan, confirmation_reserve
from .records import CandidateRow, new_candidate_row
from .results import ExperimentResult, Solution
from .scoring import (
    BicScorer,
    CompactScorer,
    DensityScorer,
    FuzzyScorer,
    HeldoutLikelihoodScorer,
    HoldoutSplit,
    ObjectiveScorer,
    make_holdout_split,
)
from .selection import (
    build_shortlist,
    choose,
    deduplicate,
    exploration_rejections,
    quality,
)
from .structure import StructureTester, null_grid, structure_reserve

__all__ = ["AutoCluster", "candidate_id", "stable_seed"]

logger = logging.getLogger(__name__)

_MAX_SEED: int = 2**32
_ERROR_POLICIES: tuple[str, ...] = ("record", "raise")
_FITTED_ATTRIBUTES: tuple[str, ...] = (
    "result_",
    "results_",
    "n_features_in_",
    "feature_names_in_",
)
_LIMITATIONS: tuple[str, ...] = (
    "Stability thresholds are engineering rules checked on a synthetic "
    "benchmark; the null-reference test bounds false recommendations only "
    "when the data follow one of its reference families.",
    "Soft deadline checked between fits; preventive memory estimate, not an "
    "RSS cap.",
    "Confirmation compares each refit to the exploratory full-data "
    "partition; not independent validation.",
    "DBCV uses the fixed metric sample and known Euclidean dimension; "
    "sampled scores are approximate diagnostics.",
)


def _finite(value: Any) -> bool:
    """Whether ``value`` is a finite number."""
    return value is not None and bool(np.isfinite(value))


class AutoCluster(ClusterMixin, BaseEstimator):
    """Clustering search with explicit geometry and stability evidence.

    ``fit(X, y)`` ignores ``y``; no universal score is defined. Each
    objective has its own primary metric:

    - ``compact``: fixed-reference silhouette plus two-axis stability.
    - ``probabilistic``: comparable BIC (GMM) or reserved held-out
      likelihood, plus confirmation on the same representation.
    - ``density``: reference-space DBCV (optional backend), coverage and
      confirmation.
    - ``fuzzy``: fixed-fuzzifier generalised Xie-Beni (optional backend)
      and confirmation.

    Parameters
    ----------
    schema : Schema or None
        Column roles; inferred from dtypes when ``None``.
    geometry : GeometrySpec or None
        Reference geometry; Gower when any feature is nominal, ordinal or
        binary, standardised euclidean otherwise.
    objective : {"compact", "probabilistic", "density", "fuzzy"}
        Objective whose primary metric ranks the candidates.
    selection : SelectionPolicy or None
        Selection thresholds; the P1 defaults when ``None``.
    budget : SearchBudget or None
        Fit, time and memory limits; defaults when ``None``.
    candidates : sequence of CandidateSpec or None
        Explicit candidates (expert mode).
    k_range : sequence of int
        Cluster counts for automatic grids.
    random_state : int
        Base seed in ``[0, 2**32)``; every seed derives from it.
    n_jobs : int
        Native thread limit per fit. The search itself is sequential.
    error_policy : {"record", "raise"}
        Record backend errors per candidate or raise them (debugging).
    probabilistic_metric : {"bic", "heldout_loglikelihood"}
        Primary metric of the probabilistic objective.
    validation_fraction : float
        Reserved fraction for ``heldout_loglikelihood``, in ``(0, 0.5)``.
    families : sequence of str or None
        Families of an automatic grid (exclusive with ``candidates``).

    Attributes
    ----------
    result_ : ExperimentResult
        Status, evidence and every fitted solution.
    results_ : pandas.DataFrame
        Per-candidate table (same as ``result_.candidates_table()``).
    n_features_in_ : int
        Number of input columns seen during ``fit``.
    feature_names_in_ : numpy.ndarray
        Column names, when all of them are strings.
    """

    def __init__(
        self,
        schema: Schema | None = None,
        geometry: GeometrySpec | None = None,
        objective: Objective = "compact",
        selection: SelectionPolicy | None = None,
        budget: SearchBudget | None = None,
        candidates: Any = None,
        k_range: Any = (2, 3, 4, 5, 6, 7, 8),
        random_state: int = 42,
        n_jobs: int = 1,
        error_policy: str = "record",
        probabilistic_metric: str = "bic",
        validation_fraction: float = 0.2,
        families: Any = None,
    ) -> None:
        self.schema = schema
        self.geometry = geometry
        self.objective = objective
        self.selection = selection
        self.budget = budget
        self.candidates = candidates
        self.k_range = k_range
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.error_policy = error_policy
        self.probabilistic_metric = probabilistic_metric
        self.validation_fraction = validation_fraction
        self.families = families

    def plan(self, X: Any) -> dict[str, Any]:
        """Validate the configuration and return the planned candidates.

        No transformation or model is fitted. Pairwise memory is guarded at
        execution time.
        """
        plan = self._prepare(X)
        holdout = self._uses_holdout()
        return {
            "candidates": [
                {"candidate_id": candidate_id(s), **candidate_parameters(s)}
                for s in plan.candidates
            ],
            "exclusions": list(plan.exclusions),
            "geometry": asdict(plan.geometry),
            "schema": asdict(plan.schema),
            "diagnostics": plan.diagnostics,
            "rows": len(plan.frame),
            "maximum_confirmation_fits": confirmation_reserve(
                plan.candidates, plan.policy
            ),
            "reserved_structure_fits": structure_reserve(
                plan.candidates, plan.policy, self.objective
            ),
            "max_fits": plan.budget.max_fits,
            "exploration_fits_per_candidate": 2 if holdout else 1,
            "note": (
                "Validation only; no fitted statistics are retained. "
                "Pairwise memory is guarded at execution."
            ),
        }

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None) -> Self:
        """Run the search and store the result.

        Fitted attributes are cleared first and only set when the search
        completes, so a failed call never leaves a stale success behind.

        Raises
        ------
        CapabilityError
            If ``sample_weight`` is given or no candidate is compatible.
        SearchFailedError
            If every attempted candidate failed.
        """
        for name in _FITTED_ATTRIBUTES:
            self.__dict__.pop(name, None)
        if sample_weight is not None:
            raise CapabilityError(
                "Sample weights are not implemented end-to-end."
            )
        if y is not None:
            warnings.warn(
                "y is ignored; use result_.external_evaluate(y) after "
                "fitting.",
                UserWarning,
                stacklevel=2,
            )
        plan = self._prepare(X)
        run = _SearchRun(
            plan=plan,
            objective=self.objective,
            probabilistic_metric=self.probabilistic_metric,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            error_policy=self.error_policy,
        )
        result = run.execute()
        self.result_ = result
        self.results_ = result.candidates_table()
        self.n_features_in_ = plan.frame.shape[1]
        if all(isinstance(c, str) for c in plan.frame.columns):
            self.feature_names_in_ = np.asarray(
                plan.frame.columns, dtype=object
            )
        if run.all_failed:
            raise SearchFailedError(
                "Every attempted candidate failed; inspect exception.result.",
                result,
            )
        return self

    def summary(self) -> dict[str, Any]:
        """Short report of a fitted experiment (see ``result_.summary``)."""
        check_is_fitted(self, "result_")
        return self.result_.summary()

    def report(self, *, top: int = 5) -> str:
        """Readable Markdown of the fitted experiment."""
        check_is_fitted(self, "result_")
        return self.result_.report(top=top)

    def ranking_table(self, top: int | None = 15) -> Any:
        """Readable candidate comparison, without refitting."""
        check_is_fitted(self, "result_")
        return self.result_.ranking_table(top=top)

    def fit_predict(self, X: Any, y: Any = None, **kwargs: Any) -> np.ndarray:
        """Fit and return the recommended labels.

        Raises
        ------
        NoRecommendationError
            If the search has no confirmed recommendation.
        """
        return self.fit(X, y=y, **kwargs).labels_

    def predict(self, X: Any) -> np.ndarray:
        """Assign new rows with the recommended pipeline, without refitting.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            New rows with exactly the training columns, in order.

        Raises
        ------
        NoRecommendationError
            If the search has no confirmed recommendation.
        CapabilityError
            If the recommended adapter is transductive.
        ValueError
            If the rows violate the training schema.
        """
        return self._recommended().assign(X)

    @property
    def labels_(self) -> np.ndarray:
        """Training labels of the recommended solution."""
        return self._recommended().labels_

    @property
    def best_estimator_(self) -> FittedCandidate:
        """Fitted pipeline of the recommended solution."""
        return self._recommended().pipeline_

    def _recommended(self) -> Solution:
        check_is_fitted(self, "result_")
        recommendation = self.result_.recommendation
        if recommendation is None:
            raise NoRecommendationError(self.result_)
        return self.result_.select(recommendation.candidate_id)

    def _uses_holdout(self) -> bool:
        return (
            self.objective == "probabilistic"
            and self.probabilistic_metric == "heldout_loglikelihood"
        )

    def _validate_execution_parameters(self) -> None:
        fraction = self.validation_fraction
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float, np.floating))
            or not np.isfinite(fraction)
            or not 0 < fraction < 0.5
        ):
            raise ValueError("validation_fraction must be finite in (0, 0.5).")
        seed = self.random_state
        if not is_integer(seed) or not 0 <= int(seed) < _MAX_SEED:
            raise ValueError("random_state must be an integer in [0, 2**32).")
        if not is_integer(self.n_jobs) or int(self.n_jobs) < 1:
            raise ValueError(
                "n_jobs is the positive native thread limit; search is "
                "sequential."
            )
        if self.error_policy not in _ERROR_POLICIES:
            raise ValueError("error_policy must be record or raise.")

    def _prepare(self, X: Any) -> SearchPlan:
        self._validate_execution_parameters()
        planner = CandidatePlanner(
            objective=self.objective,
            schema=self.schema,
            geometry=self.geometry,
            selection=self.selection,
            budget=self.budget,
            candidates=self.candidates,
            families=self.families,
            k_range=self.k_range,
            probabilistic_metric=self.probabilistic_metric,
        )
        return planner.plan(X)


class _SearchRun:
    """State and steps of a single ``AutoCluster.fit`` call."""

    def __init__(
        self,
        plan: SearchPlan,
        objective: str,
        probabilistic_metric: str,
        validation_fraction: float,
        random_state: int,
        n_jobs: int,
        error_policy: str,
    ) -> None:
        self.plan = plan
        self.objective = objective
        self.probabilistic_metric = probabilistic_metric
        self.random_state = int(random_state)
        self.error_policy = error_policy
        self.use_holdout = (
            objective == "probabilistic"
            and probabilistic_metric == "heldout_loglikelihood"
        )
        self.executor = FitExecutor(
            plan.geometry, plan.schema, plan.budget, self.random_state,
            int(n_jobs),
        )
        self.rows: list[CandidateRow] = []
        self.solutions: dict[str, Solution] = {}
        self.interrupted = plan.truncated
        self.incomplete_reasons: list[str] = []
        self._prepare_reference()
        self.split: HoldoutSplit | None = None
        if self.use_holdout:
            self.split = make_holdout_split(
                len(plan.frame),
                validation_fraction,
                stable_seed(self.random_state, "common", "likelihood-split"),
            )
        self.scorer = self._make_scorer()
        self.structure_record: dict[str, Any] | None = None
        self.gap_record: list[dict[str, Any]] | None = None
        self.selected_k: int | None = None
        self.selected_k_at_boundary: bool = False
        self.selection_warnings: list[str] = []
        self.structure_seconds = 0.0

    @property
    def all_failed(self) -> bool:
        """Whether candidates were attempted and none produced a fit."""
        return bool(self.rows) and not self.solutions and not self.interrupted

    def execute(self) -> ExperimentResult:
        """Explore, confirm, select and assemble the result."""
        self._explore()
        if self.objective == "compact" and self.plan.policy.uses_gap:
            shortlist, winner, frontier, admissible = self._select_by_gap()
        else:
            shortlist = self._shortlist()
            self._confirm(shortlist)
            for row in self.rows:
                row["fits_used"] = self.executor.fits_for(
                    row["candidate_id"]
                )
            winner, frontier, admissible = choose(
                self.rows, self.objective, self.plan.policy
            )
            if structure_reserve(
                self.plan.candidates, self.plan.policy, self.objective
            ):
                winner, admissible = self._test_structure(winner, admissible)
        common_ids = self._common_support(shortlist)
        status = self._status(winner, admissible)
        if status != "recommended":
            winner = None
        logger.info("Search finished with status %r after %d fits.",
                    status, self.executor.fits_used)
        metadata = self._metadata(admissible, shortlist, common_ids)
        return ExperimentResult(
            status, self.rows, self.solutions, winner, metadata, frontier
        )

    def _prepare_reference(self) -> None:
        """Fit the common reference geometry on a fixed metric sample."""
        plan = self.plan
        n_rows = len(plan.frame)
        rng = np.random.default_rng(
            stable_seed(self.random_state, "common", "metric")
        )
        size = min(n_rows, plan.budget.metric_sample_size)
        self.metric_indices = np.sort(rng.choice(n_rows, size, replace=False))
        self.reference = ReferenceGeometry(plan.geometry, plan.schema).fit(
            plan.frame
        )
        guard_pairwise(len(self.metric_indices), plan.budget.max_pairwise_mb)
        self.metric_values = self.reference.transform(
            plan.frame.iloc[self.metric_indices]
        )
        self.distances = self.reference.distances(self.metric_values)

    def _make_scorer(self) -> ObjectiveScorer:
        if self.objective == "density":
            return DensityScorer(
                self.metric_values, self.metric_indices,
                self.plan.geometry.kind, self.plan.budget,
            )
        if self.objective == "fuzzy":
            return FuzzyScorer()
        if self.split is not None:
            return HeldoutLikelihoodScorer(
                self.executor, self.plan.frame, self.split, self.error_policy
            )
        if self.objective == "probabilistic":
            return BicScorer()
        return CompactScorer()

    def _explore(self) -> None:
        """Fit and score each planned candidate once on all rows."""
        reserve = confirmation_reserve(
            self.plan.candidates, self.plan.policy
        ) + structure_reserve(
            self.plan.candidates, self.plan.policy, self.objective
        )
        per_candidate = 2 if self.use_holdout else 1
        budget = self.plan.budget
        slots = min(max(per_candidate, budget.max_fits - reserve),
                    budget.max_fits)
        for spec in self.plan.candidates:
            over_slots = self.executor.fits_used + per_candidate > slots
            if not self.executor.available() or over_slots:
                self.interrupted = True
                break
            self.rows.append(self._explore_one(spec))
        self._warn_failed_families()

    def _warn_failed_families(self) -> None:
        """Warn when every explored candidate of an algorithm failed.

        The search then continues with the remaining families and would
        otherwise report a recommendation drawn from fewer algorithms
        without saying so (for instance dense-matrix families above the
        memory guard).
        """
        by_algorithm: dict[str, list[CandidateRow]] = {}
        for row in self.rows:
            by_algorithm.setdefault(row["algorithm"], []).append(row)
        for algorithm, rows in by_algorithm.items():
            if any(r["status"] == "fitted" for r in rows):
                continue
            first = next((r["error"] for r in rows if r["error"]), "")
            advice = ""
            if str(first).startswith("MemoryBudgetError"):
                advice = (" Raise SearchBudget(max_pairwise_mb) or use "
                          "sampled medoids (CandidateSpec('clara', k)).")
            message = (
                f"All {len(rows)} {algorithm} candidates failed "
                f"({first}); the search continued without that "
                f"family.{advice}"
            )
            self.selection_warnings.append(message)
            logger.warning(message)

    def _explore_one(self, spec: CandidateSpec) -> CandidateRow:
        identifier = candidate_id(spec)
        started = self.executor.elapsed()
        row = new_candidate_row(
            spec, identifier, self.executor.seed(identifier, "exploration")
        )
        try:
            fitted, notices = self.executor.run(
                spec, self.plan.frame, "exploration", "exploration"
            )
            row.update(partition_metrics(
                fitted.labels_, self.distances, self.metric_indices
            ))
            row.update(
                status="fitted",
                warnings=notices,
                bic=fitted.bic_,
                converged=fitted.converged_,
                backend_parameters=fitted.backend_parameters_,
                fuzzy_diagnostics=fitted.fuzzy_diagnostics_,
            )
            self.solutions[identifier] = Solution(
                identifier, fitted, self.plan.frame, self.plan.schema
            )
            if self.scorer.score(row, fitted, spec):
                self.interrupted = True
            row["rejection_reasons"].extend(exploration_rejections(
                row, fitted.converged_, self.plan.policy, self.objective,
                len(self.plan.frame),
            ))
        except BACKEND_ERRORS as exc:
            if self.error_policy == "raise":
                raise
            row.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                warnings=list(self.executor.last_notices),
            )
            self.solutions.pop(identifier, None)
            logger.debug("Candidate %s failed: %s", identifier, exc)
        row["fit_seconds"] = self.executor.elapsed() - started
        return row

    def _select_by_gap(
        self,
    ) -> tuple[list[CandidateRow], str | None, list[str], list[str]]:
        """Null test, gap choice of k, then confirmation at that k only.

        1. The best explored silhouette is tested against the null
           reference (every explored candidate refitted on each replicate).
           Failing abstains before any confirmation fit is spent.
        2. The number of clusters is chosen by the gap statistic computed
           from the same null fits (one-standard-error rule).
        3. Only candidates with that ``k`` are confirmed by stability; the
           usual admissibility and preference rules pick among them.

        Returns
        -------
        tuple
            Shortlist, winner, Pareto frontier and admissible identifiers.
        """
        explored = [
            r for r in self.rows if r["candidate_id"] in self.solutions
        ]
        scored = [r for r in explored if _finite(r["primary_score"])]
        if not scored:
            return [], None, [], []
        top = max(scored, key=lambda r: quality(r, self.objective))
        tester = StructureTester(
            self.executor, self.plan.frame, self.plan.schema,
            self.plan.geometry, self.metric_indices, self.plan.policy,
            self.plan.budget.max_pairwise_mb, self.error_policy,
        )
        grid = [self.solutions[r["candidate_id"]].pipeline_.spec
                for r in explored]
        started = time.perf_counter()
        outcome = tester.test(quality(top, self.objective), grid)
        self.structure_seconds = time.perf_counter() - started
        top["structure_test"] = outcome.as_record()
        self.structure_record = outcome.as_record()
        if outcome.exhausted or outcome.failed:
            if outcome.exhausted:
                self.interrupted = True
            self.incomplete_reasons.append(
                outcome.reason or "null reference test inconclusive"
            )
            self._refresh_fit_counts()
            return [], None, [], []
        if not outcome.passed:
            for row in scored:
                row["rejection_reasons"].append(str(outcome.reason))
            self._refresh_fit_counts()
            return [], None, [], []
        table = gap_table(self.rows, outcome.null_dispersions)
        self.gap_record = [entry.as_record() for entry in table]
        observed_ks = {
            int(row["parameters"]["k"]) for row in scored
        }
        available_ks = {entry.k for entry in table}
        if not table or available_ks != observed_ks:
            self.incomplete_reasons.append(
                "gap statistic unavailable for one or more explored k values"
            )
            self._refresh_fit_counts()
            return [], None, [], []
        chosen_k = select_k(table)
        self.selected_k = chosen_k
        self.selected_k_at_boundary = (
            len(table) > 1 and chosen_k == table[-1].k
        )
        if self.selected_k_at_boundary:
            message = (
                f"gap statistic selected k={chosen_k}, the largest explored "
                "value; the true number of clusters may be larger: widen "
                "k_range"
            )
            self.selection_warnings.append(message)
            logger.warning(message)
            if self.plan.policy.boundary_k_policy == "abstain":
                # Governance decision, not a change of the statistic: the
                # curve and the diagnostic k are kept, but an unresolved
                # upper bound is never promoted to a validated result.
                self.incomplete_reasons.append(
                    f"gap statistic reached the largest explored k="
                    f"{chosen_k} and boundary_k_policy='abstain' withheld "
                    "confirmation; widen k_range to resolve"
                )
                for row in scored:
                    row["rejection_reasons"].append(
                        "confirmation withheld at the k_range boundary"
                    )
                self._refresh_fit_counts()
                return [], None, [], []
        for row in scored:
            if row["parameters"]["k"] != chosen_k:
                row["rejection_reasons"].append(
                    f"gap statistic selects k={chosen_k}"
                )
        # Candidates sharing k are few, so all are confirmed without
        # deduplication: two identical partitions can differ in stability
        # (a deterministic adapter versus a local search), and dropping the
        # stable twin before confirmation would lose the solution.
        shortlist = self._shortlist(deduplicated=False)
        self._confirm(shortlist)
        self._refresh_fit_counts()
        winner, frontier, admissible = choose(
            self.rows, self.objective, self.plan.policy
        )
        return shortlist, winner, frontier, admissible

    def _refresh_fit_counts(self) -> None:
        for row in self.rows:
            row["fits_used"] = self.executor.fits_for(row["candidate_id"])

    def _shortlist(self, deduplicated: bool = True) -> list[CandidateRow]:
        """Fix the diverse confirmation shortlist before any replica."""
        eligible = [
            r for r in self.rows
            if r["status"] == "fitted" and not r["rejection_reasons"]
        ]
        eligible.sort(
            key=lambda r: (-quality(r, self.objective), r["candidate_id"])
        )
        diverse = (
            deduplicate(
                eligible, self.solutions, self.plan.policy.duplicate_ari
            )
            if deduplicated else eligible
        )
        return build_shortlist(diverse, self.plan.policy.shortlist_size)

    def _test_structure(
        self, winner: str | None, admissible: list[str]
    ) -> tuple[str | None, list[str]]:
        """Test the preferred candidate's family against the null reference.

        A single test is run. Its statistic is the best silhouette of the
        null grid (every explored candidate by default), the same
        statistic computed on every null replicate. If it fails, no other
        candidate is tried: retrying would give several chances to beat
        the reference and void its error bound, so the search abstains.
        """
        if winner is None:
            return winner, admissible
        tester = StructureTester(
            self.executor, self.plan.frame, self.plan.schema,
            self.plan.geometry, self.metric_indices, self.plan.policy,
            self.plan.budget.max_pairwise_mb, self.error_policy,
        )
        spec = self.solutions[winner].pipeline_.spec
        explored = [
            self.solutions[r["candidate_id"]].pipeline_.spec
            for r in self.rows if r["candidate_id"] in self.solutions
        ]
        grid = null_grid(spec, explored, self.plan.policy.null_scope)
        members = {candidate_id(s) for s in grid}
        scores = [
            quality(r, self.objective) for r in self.rows
            if r["candidate_id"] in members and _finite(r["primary_score"])
        ]
        outcome = tester.test(max(scores), grid)
        self.structure_record = outcome.as_record()
        rows = {r["candidate_id"]: r for r in self.rows}
        rows[winner]["structure_test"] = outcome.as_record()
        for candidate in self.rows:
            candidate["fits_used"] = self.executor.fits_for(
                candidate["candidate_id"]
            )
        if outcome.passed:
            return winner, admissible
        if outcome.exhausted or outcome.failed:
            if outcome.exhausted:
                self.interrupted = True
            self.incomplete_reasons.append(
                outcome.reason or "null reference test inconclusive"
            )
            return None, admissible
        for identifier in admissible:
            rows[identifier]["rejection_reasons"].append(
                outcome.reason if identifier in members else
                "not established: the preferred family did not exceed the "
                "null reference"
            )
        return None, []

    def _confirm(self, shortlist: list[CandidateRow]) -> None:
        confirmer = StabilityConfirmer(
            self.executor, self.plan.frame, self.plan.policy,
            self.error_policy,
        )
        for row in shortlist:
            if confirmer.confirm(row, self.solutions[row["candidate_id"]]):
                self.interrupted = True

    def _common_support(self, shortlist: list[CandidateRow]) -> np.ndarray:
        """Secondary silhouette on the shortlist's common assigned rows.

        This diagnostic never replaces the primary metric.
        """
        if not shortlist:
            return np.array([], dtype=int)
        common = np.ones(len(self.metric_indices), dtype=bool)
        for row in shortlist:
            labels = self.solutions[row["candidate_id"]].labels_
            common &= labels[self.metric_indices] >= 0
        common_ids = self.metric_indices[common]
        common_distances = self.distances[np.ix_(common, common)]
        for row in shortlist:
            metric = partition_metrics(
                self.solutions[row["candidate_id"]].labels_,
                common_distances,
                common_ids,
            )
            row.update(
                common_support_count=len(common_ids),
                silhouette_common_support=metric["silhouette"],
                silhouette_common_reason=metric["silhouette_reason"],
            )
        return common_ids

    def _status(self, winner: str | None, admissible: list[str]) -> str:
        if any(
            (r["validation_diagnostics"] or {}).get("error") for r in self.rows
        ):
            self.incomplete_reasons.append(
                "one or more primary validation fits failed"
            )
        if any(r["primary_unavailable"] for r in self.rows):
            self.incomplete_reasons.append(
                "density primary metric unavailable for the requested "
                "geometry/backend"
            )
        if self.interrupted:
            self.incomplete_reasons.append(
                "search grid or confirmation truncated by candidate/fit/time "
                "budget"
            )
        # A partial protocol can never yield a validated recommendation.
        if self.incomplete_reasons:
            return "incomplete"
        if winner:
            return "recommended"
        return "ambiguous" if admissible else "no_admissible_solution"

    def _metadata(
        self,
        admissible: list[str],
        shortlist: list[CandidateRow],
        common_ids: np.ndarray,
    ) -> dict[str, Any]:
        plan = self.plan
        split = self.split
        gower = plan.geometry.kind == "gower"
        return {
            "package_version": __version__,
            "methodology": plan.policy.version,
            "objective": self.objective,
            "probabilistic_metric": (
                self.probabilistic_metric
                if self.objective == "probabilistic" else None
            ),
            "validation_indices": (
                split.validation.tolist() if split is not None else None
            ),
            "training_indices": (
                split.training.tolist() if split is not None else None
            ),
            "incomplete_reasons": list(self.incomplete_reasons),
            "admissible_candidates": admissible,
            "fit_breakdown": dict(self.executor.breakdown),
            "common_support_indices": common_ids.tolist(),
            "geometry": asdict(plan.geometry),
            "schema": asdict(plan.schema),
            "diagnostics": plan.diagnostics,
            "policy": asdict(plan.policy),
            "budget": asdict(plan.budget),
            "exclusions": list(plan.exclusions),
            "fits_used": self.executor.fits_used,
            "elapsed_seconds": self.executor.elapsed(),
            "random_state": self.random_state,
            "native_thread_limit": self.executor.thread_limit,
            "parallel_workers": self.executor.workers_used,
            "stage_seconds": {
                k: round(v, 3)
                for k, v in self.executor.stage_seconds.items()
            },
            "structure_test_seconds": round(self.structure_seconds, 3),
            "metric_indices": self.metric_indices.tolist(),
            "reference_features": list(self.reference.columns_),
            "reference_weights": (
                (dict(plan.geometry.weights) or
                 "equal per nonconstant original variable") if gower
                else "unit after declared scaling"
            ),
            "confirmation_candidates": [r["candidate_id"] for r in shortlist],
            "k_selection": (
                "gap" if self.objective == "compact"
                and plan.policy.uses_gap else "silhouette"
            ),
            "structure_test": self.structure_record,
            "validation_scope": (
                "reference_screen_and_stability"
                if self.objective == "compact"
                and plan.policy.null_replicates > 0
                else "stability_only"
                if self.objective == "compact"
                else "objective_score_and_stability_no_structure_test"
            ),
            "gap_table": self.gap_record,
            "selected_k": self.selected_k,
            "selected_k_at_boundary": self.selected_k_at_boundary,
            "boundary_k_policy": plan.policy.boundary_k_policy,
            "null_family": (
                self.structure_record["family"]
                if self.structure_record else None
            ),
            "selection_warnings": list(self.selection_warnings),
            "environment": dependency_versions(),
            "limitations": list(_LIMITATIONS),
        }
