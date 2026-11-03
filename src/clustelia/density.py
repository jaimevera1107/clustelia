"""DBCV delegated to a pinned optional backend, in the reference space.

Conservative support rules: at least two sampled clusters with three
observations each and no duplicate reference points (no jitter is added).
Gower is not evaluated because no valid euclidean dimension is
established for that geometry. A missing score is never replaced by zero.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .geometry import GeometryValues, guard_pairwise
from .optional import OPTIONALS, optional_version, require_optional

__all__ = ["DBCV_MIN_CLUSTER_SIZE", "DBCV_PAIRWISE_MATRICES",
           "density_validity"]

DBCV_MIN_CLUSTER_SIZE: int = 3
"""Minimum sampled observations per assigned cluster."""

DBCV_PAIRWISE_MATRICES: int = 9
"""Conservative count of simultaneous pairwise arrays used by the backend."""

_SCORE_EPSILON: float = 1e-10


def _backend_label() -> str:
    pin = OPTIONALS["density"].required_version
    return f"hdbscan.validity.validity_index@{pin}"


def density_validity(
    values: GeometryValues,
    labels: np.ndarray,
    geometry_kind: str,
    max_pairwise_mb: float,
) -> dict[str, Any]:
    """Compute DBCV for a partition of the metric sample.

    Parameters
    ----------
    values : array-like of shape (m, d)
        Reference representation of the metric sample.
    labels : numpy.ndarray of shape (m,)
        Labels of the metric sample; negative values denote noise.
    geometry_kind : str
        Reference geometry kind; only ``"euclidean"`` is evaluated.
    max_pairwise_mb : float
        Preventive pairwise memory budget.

    Returns
    -------
    dict
        ``dbcv`` (or ``None``), ``dbcv_reason``, ``dbcv_backend``,
        ``dbcv_warnings``, ``dbcv_support`` and ``primary_unavailable``.
    """
    labels = np.asarray(labels)
    result: dict[str, Any] = {
        "dbcv": None,
        "dbcv_reason": None,
        "dbcv_backend": None,
        "dbcv_warnings": [],
        "dbcv_support": len(labels),
        "primary_unavailable": False,
    }
    unavailable = _unavailable_reason(geometry_kind)
    if unavailable is not None:
        result.update(dbcv_reason=unavailable, primary_unavailable=True)
        return result
    result["dbcv_backend"] = _backend_label()
    groups, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(groups) < 2 or np.min(counts) < DBCV_MIN_CLUSTER_SIZE:
        result["dbcv_reason"] = (
            "DBCV requires >=2 sampled assigned clusters and "
            f">={DBCV_MIN_CLUSTER_SIZE} observations in each"
        )
        return result
    points = np.asarray(values, dtype=np.float64)
    if len(np.unique(points, axis=0)) != len(points):
        result["dbcv_reason"] = (
            "Duplicate reference points are not supported by the DBCV "
            "protocol; no jitter is applied"
        )
        return result
    result.update(_score(points, _encode(labels, groups), max_pairwise_mb))
    return result


def _unavailable_reason(geometry_kind: str) -> str | None:
    """Reason why DBCV cannot be the primary metric, if any."""
    if geometry_kind != "euclidean":
        return (
            "DBCV requires numeric euclidean geometry with known dimension"
        )
    if optional_version("density") is None:
        pin = OPTIONALS["density"].required_version
        return (
            "Install clustelia[density] for "
            f"hdbscan.validity.validity_index=={pin}"
        )
    return None


def _encode(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Relabel clusters to 0..k-1 while preserving -1 noise."""
    encoded = np.full(len(labels), -1, dtype=int)
    for position, label in enumerate(groups):
        encoded[labels == label] = position
    return encoded


def _score(
    points: np.ndarray, encoded: np.ndarray, max_pairwise_mb: float
) -> dict[str, Any]:
    """Call the backend and validate its output domain."""
    try:
        guard_pairwise(len(encoded), max_pairwise_mb, DBCV_PAIRWISE_MATRICES)
        require_optional("density")
        from hdbscan.validity import validity_index

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score = float(
                validity_index(
                    points, encoded, metric="euclidean", mst_raw_dist=False
                )
            )
    except (ValueError, RuntimeError, FloatingPointError,
            MemoryError) as exc:
        return {
            "dbcv_reason": (
                "DBCV unavailable for this candidate: "
                f"{type(exc).__name__}: {exc}"
            )
        }
    output: dict[str, Any] = {
        "dbcv_warnings": [str(w.message) for w in caught]
    }
    lower, upper = -1 - _SCORE_EPSILON, 1 + _SCORE_EPSILON
    if not np.isfinite(score) or not lower <= score <= upper:
        output["dbcv_reason"] = (
            "DBCV backend returned a non-finite or out-of-domain score"
        )
    else:
        output["dbcv"] = score
    return output
