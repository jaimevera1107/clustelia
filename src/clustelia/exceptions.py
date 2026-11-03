"""Public exception hierarchy.

Every exception raised deliberately by clustelia derives from
:class:`ClusteliaError`, so callers can catch library errors without
catching unrelated failures. Each class also derives from the closest
built-in exception to stay compatible with conventional ``except`` blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .results import ExperimentResult

__all__ = [
    "CapabilityError",
    "ClusteliaError",
    "MemoryBudgetError",
    "NoRecommendationError",
    "SearchFailedError",
]


class ClusteliaError(Exception):
    """Base class for all errors raised deliberately by clustelia."""


class CapabilityError(ClusteliaError, ValueError):
    """The requested combination has no implemented adapter or backend."""


class MemoryBudgetError(ClusteliaError, MemoryError):
    """A preventive memory estimate exceeds the configured budget.

    The check is an estimate computed before allocating, not a measurement
    of the resident memory of the process.
    """


class SearchFailedError(ClusteliaError, RuntimeError):
    """Every attempted candidate failed; the partial result is attached.

    Parameters
    ----------
    message : str
        Human-readable description of the failure.
    result : ExperimentResult
        Result with the per-candidate errors and warnings.
    """

    def __init__(self, message: str, result: ExperimentResult) -> None:
        super().__init__(message)
        self.result = result


class NoRecommendationError(ClusteliaError, RuntimeError, AttributeError):
    """The search finished without a confirmed recommendation.

    It derives from :class:`AttributeError` so that ``hasattr(model,
    "labels_")`` returns ``False`` instead of raising, mirroring the
    behaviour of scikit-learn's ``NotFittedError``.

    Parameters
    ----------
    result : ExperimentResult
        Result to inspect for alternatives and rejection reasons.
    """

    def __init__(self, result: ExperimentResult) -> None:
        super().__init__(
            "No confirmed recommendation: "
            f"{result.selection_status}. Inspect result_."
        )
        self.result = result
