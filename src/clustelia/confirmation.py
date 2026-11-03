"""Stability confirmation of the fixed shortlist.

Every replica refits the complete pipeline (scaling and PCA included).
Replicas are compared with the exploratory full-data partition, which is
not an independent validation and does not remove selection bias.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from .adapters import REGISTRY
from .config import SelectionPolicy
from .evaluation import stability_pair, summarize_stability
from .execution import BACKEND_ERRORS, FitExecutor
from .records import CandidateRow
from .results import Solution

__all__ = ["StabilityConfirmer"]

_MIN_SUBSAMPLE_ROWS: int = 2


class StabilityConfirmer:
    """Run the composition and seed axes for shortlisted candidates.

    Subsamples are common to all candidates (derived from ``"common"``
    seeds); replica seeds derive from the candidate and the phase.

    Parameters
    ----------
    executor : FitExecutor
        Executor charging every replica to the budget.
    frame : pandas.DataFrame
        Training table.
    policy : SelectionPolicy
        Replica counts and acceptance thresholds.
    error_policy : {"record", "raise"}
        Whether replica failures are recorded or raised.
    """

    def __init__(
        self,
        executor: FitExecutor,
        frame: pd.DataFrame,
        policy: SelectionPolicy,
        error_policy: str,
    ) -> None:
        self.executor = executor
        self.frame = frame
        self.policy = policy
        self.error_policy = error_policy
        self._subsample = cache(self._draw_subsample)

    def confirm(self, row: CandidateRow, solution: Solution) -> bool:
        """Confirm one candidate in place.

        Returns
        -------
        bool
            ``True`` if the budget interrupted the confirmation.
        """
        spec = solution.pipeline_.spec
        axes: list[tuple[str, int]] = [
            ("composition", self.policy.subsamples)
        ]
        if REGISTRY[spec.algorithm].stochastic:
            axes.append(("randomness", self.policy.seed_repeats))
        interrupted = False
        for axis, repeats in axes:
            records: list[dict[str, Any]] = []
            for replica in range(repeats):
                if not self.executor.available():
                    interrupted = True
                    break
                records.append(self._replicate(solution, axis, replica))
            row[axis] = dict(
                **summarize_stability(records, repeats), records=records
            )
        row["confirmed"] = self._complete(row)
        if not row["confirmed"]:
            row["rejection_reasons"].append(
                "confirmation incomplete within budget"
            )
        return interrupted

    def _complete(self, row: CandidateRow) -> bool:
        composition_done = (
            row["composition"]["completed"] == self.policy.subsamples
        )
        randomness = row["randomness"]
        randomness_done = (
            randomness is None
            or randomness["completed"] == self.policy.seed_repeats
        )
        return bool(composition_done and randomness_done)

    def _draw_subsample(self, replica: int) -> np.ndarray:
        n_rows = len(self.frame)
        seed = self.executor.seed("common", f"subsample-{replica}")
        rng = np.random.default_rng(seed)
        size = max(
            _MIN_SUBSAMPLE_ROWS, int(n_rows * self.policy.sample_fraction)
        )
        return np.sort(rng.choice(n_rows, size, replace=False))

    def _replicate(
        self, solution: Solution, axis: str, replica: int
    ) -> dict[str, Any]:
        """Fit one replica and compare it with the exploratory labels."""
        if axis == "composition":
            indices = self._subsample(replica)
        else:
            indices = np.arange(len(self.frame))
        phase = f"confirmation-{axis}-{replica}"
        try:
            fitted, notices = self.executor.run(
                solution.pipeline_.spec, self.frame.iloc[indices], phase, axis
            )
        except BACKEND_ERRORS as exc:
            if self.error_policy == "raise":
                raise
            return {
                "ari": None,
                "error": f"{type(exc).__name__}: {exc}",
                "warnings": list(self.executor.last_notices),
            }
        comparison = stability_pair(
            solution.labels_[indices], fitted.labels_, self.policy.min_common
        )
        reasons = self._replica_reasons(fitted.converged_, fitted.labels_,
                                        comparison)
        if reasons:
            comparison.update(ari=None, reason="; ".join(reasons))
        comparison.update(
            seed=self.executor.seed(solution.candidate_id, phase),
            warnings=notices,
        )
        return comparison

    def _replica_reasons(
        self,
        converged: bool,
        labels: np.ndarray,
        comparison: dict[str, Any],
    ) -> list[str]:
        """Reasons a replica cannot contribute an evaluable ARI."""
        reasons: list[str] = []
        if not converged:
            reasons.append("replica did not converge")
        if np.mean(labels >= 0) < self.policy.min_coverage:
            reasons.append("replica below coverage threshold")
        agreement = comparison["assignment_agreement"]
        if agreement < self.policy.min_assignment_agreement:
            reasons.append(
                "replica assignment/noise agreement below threshold"
            )
        return reasons
