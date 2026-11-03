"""Budgeted execution of pipeline fits.

:class:`FitExecutor` is the only place where fits are started. It counts
every attempt (failed ones included) against ``max_fits``, enforces the
soft deadline between fits, limits native threads, derives seeds and
captures backend warnings.
"""

from __future__ import annotations

import time
import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from joblib import cpu_count
from threadpoolctl import threadpool_limits

from ._seeding import candidate_id, stable_seed
from .adapters import FittedCandidate
from .config import CandidateSpec, GeometrySpec, Schema, SearchBudget

__all__ = [
    "AUTO_MAX_WORKERS",
    "AUTO_MIN_ROWS",
    "BACKEND_ERRORS",
    "FIT_STAGES",
    "FitBudgetExhaustedError",
    "FitExecutor",
]

AUTO_MIN_ROWS: int = 1000
"""Rows below which automatic worker selection stays sequential. Measured
on 16 CPUs (Windows 11): 500 rows ran 4.0 s sequentially and 4.9 to 6.5 s
with workers (slower); 1000 rows ran 11.7 s against 6.0 s with 8 workers."""

AUTO_MAX_WORKERS: int = 8
"""Cap of the automatic worker count. Measured on 16 CPUs: 8 workers were
as fast as 16 or faster at every size, and each worker holds a data copy."""

BACKEND_ERRORS: tuple[type[BaseException], ...] = (
    ValueError,
    RuntimeError,
    MemoryError,
    FloatingPointError,
    np.linalg.LinAlgError,
)
"""Errors recorded per candidate. Anything else is a programming error and
propagates."""

FIT_STAGES: tuple[str, ...] = (
    "exploration",
    "validation",
    "composition",
    "randomness",
    "structure",
)


class FitBudgetExhaustedError(RuntimeError):
    """The fit or time budget ran out before a batch of fits finished."""


def _apply_task(
    spec: CandidateSpec,
    geometry: GeometrySpec,
    schema: Schema,
    seed: int,
    max_pairwise_mb: float,
    data: pd.DataFrame,
    thread_limit: int,
    reducer: Callable[[np.ndarray], Any],
) -> tuple[Any, BaseException | None]:
    """Fit one pipeline in a worker and reduce its labels.

    Backend errors are returned, not raised, so the parent can report the
    first one in task order whatever the completion order was.
    """
    pipeline = FittedCandidate(spec, geometry, schema, seed, max_pairwise_mb)
    try:
        with warnings.catch_warnings(), threadpool_limits(limits=thread_limit):
            warnings.simplefilter("ignore")
            pipeline.fit(data)
        return reducer(np.asarray(pipeline.labels_)), None
    except BACKEND_ERRORS as error:
        return None, error


class FitExecutor:
    """Run fits under a shared fit budget and a soft deadline.

    Parameters
    ----------
    geometry : GeometrySpec
        Reference geometry fitted inside every pipeline.
    schema : Schema
        Resolved column roles.
    budget : SearchBudget
        Fit, time and memory limits.
    random_state : int
        Base seed.
    thread_limit : int
        Native thread limit applied during each fit.
    clock : callable
        Monotonic clock in seconds (injectable for tests).
    """

    def __init__(
        self,
        geometry: GeometrySpec,
        schema: Schema,
        budget: SearchBudget,
        random_state: int,
        thread_limit: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.geometry = geometry
        self.schema = schema
        self.budget = budget
        self.random_state = random_state
        self.thread_limit = thread_limit
        self._clock = clock
        self._started = clock()
        self.fits_used = 0
        self.breakdown: dict[str, int] = dict.fromkeys(FIT_STAGES, 0)
        self._per_candidate: Counter[str] = Counter()
        self.last_notices: list[str] = []
        self.workers_used = 1
        self.stage_seconds: dict[str, float] = dict.fromkeys(FIT_STAGES, 0.0)

    def elapsed(self) -> float:
        """Seconds since the executor was created."""
        return self._clock() - self._started

    def available(self) -> bool:
        """Whether another fit may start under the fit and time budgets."""
        return (
            self.fits_used < self.budget.max_fits
            and self.elapsed() < self.budget.timeout_s
        )

    def seed(self, identifier: str, phase: str) -> int:
        """Seed of ``identifier`` in ``phase``."""
        return stable_seed(self.random_state, identifier, phase)

    def fits_for(self, identifier: str) -> int:
        """Fits attempted for one candidate across all stages."""
        return self._per_candidate[identifier]

    def run(
        self,
        spec: CandidateSpec,
        data: pd.DataFrame,
        phase: str,
        stage: str,
    ) -> tuple[FittedCandidate, list[str]]:
        """Fit one pipeline and return it with the captured warnings.

        The attempt is counted before fitting, so failures also spend
        budget. If the fit raises, the warnings emitted until the failure
        remain available in :attr:`last_notices`.

        Raises
        ------
        ValueError
            If ``stage`` is unknown.
        """
        if stage not in self.breakdown:
            raise ValueError(f"Unknown fit stage: {stage!r}.")
        identifier = candidate_id(spec)
        self.fits_used += 1
        self.breakdown[stage] += 1
        self._per_candidate[identifier] += 1
        self.last_notices = []
        started = time.perf_counter()
        pipeline = FittedCandidate(
            spec,
            self.geometry,
            self.schema,
            self.seed(identifier, phase),
            self.budget.max_pairwise_mb,
        )
        with (
            warnings.catch_warnings(record=True) as caught,
            threadpool_limits(limits=self.thread_limit),
        ):
            warnings.simplefilter("always")
            try:
                pipeline.fit(data)
            finally:
                self.last_notices = [str(w.message) for w in caught]
                self.stage_seconds[stage] += time.perf_counter() - started
        return pipeline, list(self.last_notices)

    def workers_for(self, n_tasks: int, n_rows: int) -> int:
        """Worker processes for a batch of ``n_tasks`` fits on ``n_rows``.

        Follows ``SearchBudget.workers``; ``None`` stays sequential on
        small data (below :data:`AUTO_MIN_ROWS`) and is otherwise capped
        at :data:`AUTO_MAX_WORKERS`.
        """
        requested = self.budget.workers
        if requested is None:
            if n_rows < AUTO_MIN_ROWS:
                return 1
            limit = min(AUTO_MAX_WORKERS, cpu_count())
        elif requested == -1:
            limit = cpu_count()
        else:
            limit = requested
        return max(1, min(limit, n_tasks))

    def fit_apply(
        self,
        specs: Sequence[CandidateSpec],
        data: pd.DataFrame,
        phase: str,
        stage: str,
        reducer: Callable[[np.ndarray], Any],
    ) -> list[Any]:
        """Fit every spec on ``data`` and return ``reducer(labels)`` each.

        Equivalent to calling :meth:`run` for each spec in order, with the
        same seeds, fit counts and budget semantics, but the fits may run
        in worker processes (see :meth:`workers_for`). A batch runs in
        parallel only if the fit budget covers all of it and the deadline
        has not passed; otherwise it runs sequentially and checks the
        budget before each fit. ``reducer`` must be picklable.

        Raises
        ------
        FitBudgetExhaustedError
            If the budget ran out before every fit started (sequential
            path only).
        ValueError
            If ``stage`` is unknown, or the first failing fit's error (a
            backend error is re-raised in task order; fits after it are
            not charged, as in the sequential path).
        """
        if stage not in self.breakdown:
            raise ValueError(f"Unknown fit stage: {stage!r}.")
        specs = list(specs)
        workers = self.workers_for(len(specs), len(data))
        fits_left = self.budget.max_fits - self.fits_used
        if (
            workers > 1
            and fits_left >= len(specs)
            and self.elapsed() < self.budget.timeout_s
        ):
            return self._fit_apply_parallel(
                specs, data, phase, stage, reducer, workers
            )
        results = []
        for spec in specs:
            if not self.available():
                raise FitBudgetExhaustedError(
                    "fit or time budget exhausted"
                )
            fitted, _ = self.run(spec, data, phase, stage)
            results.append(reducer(np.asarray(fitted.labels_)))
        return results

    def _fit_apply_parallel(
        self,
        specs: list[CandidateSpec],
        data: pd.DataFrame,
        phase: str,
        stage: str,
        reducer: Callable[[np.ndarray], Any],
        workers: int,
    ) -> list[Any]:
        """Run a batch in loky worker processes; count fits in order."""
        from joblib import Parallel, delayed

        identifiers = [candidate_id(spec) for spec in specs]
        started = time.perf_counter()
        outcomes = Parallel(n_jobs=workers, backend="loky")(
            delayed(_apply_task)(
                spec, self.geometry, self.schema,
                self.seed(identifier, phase), self.budget.max_pairwise_mb,
                data, self.thread_limit, reducer,
            )
            for spec, identifier in zip(specs, identifiers, strict=True)
        )
        self.stage_seconds[stage] += time.perf_counter() - started
        self.workers_used = max(self.workers_used, workers)
        self.last_notices = []
        results = []
        for identifier, (value, error) in zip(
            identifiers, outcomes, strict=True
        ):
            self.fits_used += 1
            self.breakdown[stage] += 1
            self._per_candidate[identifier] += 1
            if error is not None:
                raise error
            results.append(value)
        return results
