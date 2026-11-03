"""Objective-specific primary scores (strategy pattern).

Each objective has its own scorer. No scorer substitutes another metric
when its own is unavailable: a missing score stays ``None`` with a reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from ._seeding import candidate_id
from .adapters import FittedCandidate
from .config import CandidateSpec, SearchBudget
from .density import density_validity
from .execution import BACKEND_ERRORS, FitExecutor
from .geometry import GeometryValues
from .records import CandidateRow

__all__ = [
    "BicScorer",
    "CompactScorer",
    "DensityScorer",
    "FuzzyScorer",
    "HeldoutLikelihoodScorer",
    "HoldoutSplit",
    "ObjectiveScorer",
    "make_holdout_split",
]

_MIN_TRAINING_ROWS: int = 3
_MIN_VALIDATION_ROWS: int = 2


class ObjectiveScorer(Protocol):
    """Strategy computing the primary score of one fitted candidate."""

    def score(
        self, row: CandidateRow, fitted: FittedCandidate, spec: CandidateSpec
    ) -> bool:
        """Fill ``primary_score`` and related columns of ``row``.

        Returns
        -------
        bool
            ``True`` if the budget prevented the score from being computed.
        """
        ...


class CompactScorer:
    """Silhouette in the fixed reference geometry."""

    def score(
        self, row: CandidateRow, fitted: FittedCandidate, spec: CandidateSpec
    ) -> bool:
        """Use the reference silhouette as primary score."""
        row.update(
            primary_score=row["silhouette"],
            selection_metric="reference_silhouette",
        )
        return False


class BicScorer:
    """Negative BIC of Gaussian mixtures on a common representation."""

    def score(
        self, row: CandidateRow, fitted: FittedCandidate, spec: CandidateSpec
    ) -> bool:
        """Use ``-BIC`` as primary score (larger is better)."""
        bic = fitted.bic_
        row.update(
            primary_score=-bic if bic is not None else None,
            selection_metric="negative_bic",
        )
        return False


class FuzzyScorer:
    """Negative generalised Xie-Beni index with a fixed fuzzifier."""

    def score(
        self, row: CandidateRow, fitted: FittedCandidate, spec: CandidateSpec
    ) -> bool:
        """Use ``-XB_m`` as primary score (larger is better)."""
        diagnostics = fitted.fuzzy_diagnostics_ or {}
        xie_beni = diagnostics.get("xie_beni")
        primary = None
        if xie_beni is not None and bool(np.isfinite(xie_beni)):
            primary = -xie_beni
        row.update(
            primary_score=primary,
            selection_metric="negative_xie_beni_m",
        )
        return False


class DensityScorer:
    """DBCV on the common metric sample in the reference geometry.

    Parameters
    ----------
    metric_values : numpy.ndarray or pandas.DataFrame
        Reference representation of the metric sample.
    metric_indices : numpy.ndarray
        Positions of the metric sample among the training rows.
    geometry_kind : str
        Reference geometry kind.
    budget : SearchBudget
        Provides the pairwise memory budget.
    """

    def __init__(
        self,
        metric_values: GeometryValues,
        metric_indices: np.ndarray,
        geometry_kind: str,
        budget: SearchBudget,
    ) -> None:
        self.metric_values = metric_values
        self.metric_indices = metric_indices
        self.geometry_kind = geometry_kind
        self.budget = budget

    def score(
        self, row: CandidateRow, fitted: FittedCandidate, spec: CandidateSpec
    ) -> bool:
        """Use DBCV as primary score; record why it is missing if so."""
        row.update(
            density_validity(
                self.metric_values,
                fitted.labels_[self.metric_indices],
                self.geometry_kind,
                self.budget.max_pairwise_mb,
            )
        )
        row.update(primary_score=row["dbcv"], selection_metric="dbcv")
        if row["dbcv"] is None:
            row["rejection_reasons"].append(row["dbcv_reason"])
        return False


@dataclass(frozen=True)
class HoldoutSplit:
    """Common training/validation row positions for likelihood selection."""

    training: np.ndarray
    validation: np.ndarray


def make_holdout_split(
    n_rows: int, fraction: float, seed: int
) -> HoldoutSplit:
    """Draw a reproducible validation subset shared by all candidates.

    Raises
    ------
    ValueError
        If fewer than three training rows would remain.
    """
    rng = np.random.default_rng(seed)
    size = max(_MIN_VALIDATION_ROWS, int(n_rows * fraction))
    validation = np.sort(rng.choice(n_rows, size, replace=False))
    training = np.setdiff1d(np.arange(n_rows), validation)
    if len(training) < _MIN_TRAINING_ROWS:
        raise ValueError(
            "Insufficient training support for reserved validation."
        )
    return HoldoutSplit(training, validation)


class HeldoutLikelihoodScorer:
    """Mean validation log-likelihood of a pipeline fitted on training rows.

    The validation pipeline is a separate fit: its transformations never
    see the reserved rows. The score is in the training-fitted coordinates,
    common to all candidates of the run. It serves selection only and is
    not an unbiased final evaluation.

    Parameters
    ----------
    executor : FitExecutor
        Executor charging the validation fit to the budget.
    frame : pandas.DataFrame
        Training table.
    split : HoldoutSplit
        Common row split.
    error_policy : {"record", "raise"}
        Whether validation failures are recorded or raised.
    """

    def __init__(
        self,
        executor: FitExecutor,
        frame: pd.DataFrame,
        split: HoldoutSplit,
        error_policy: str,
    ) -> None:
        self.executor = executor
        self.frame = frame
        self.split = split
        self.error_policy = error_policy

    def score(
        self, row: CandidateRow, fitted: FittedCandidate, spec: CandidateSpec
    ) -> bool:
        """Fit on training rows and score the reserved rows."""
        if not self.executor.available():
            row["rejection_reasons"].append(
                "validation fit not completed within budget"
            )
            return True
        try:
            validator, notices = self.executor.run(
                spec,
                self.frame.iloc[self.split.training],
                "validation",
                "validation",
            )
            heldout = validator.loglikelihood(
                self.frame.iloc[self.split.validation]
            )
        except BACKEND_ERRORS as exc:
            if self.error_policy == "raise":
                raise
            row["validation_diagnostics"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "warnings": list(self.executor.last_notices),
            }
            return False
        row.update(
            heldout_loglikelihood=heldout,
            primary_score=heldout if validator.converged_ else None,
            validation_diagnostics={
                "training_count": len(self.split.training),
                "validation_count": len(self.split.validation),
                "seed": self.executor.seed(candidate_id(spec), "validation"),
                "converged": validator.converged_,
                "warnings": notices,
            },
            selection_metric="heldout_loglikelihood",
        )
        return False
