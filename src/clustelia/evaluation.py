"""Partition metrics and stability comparisons.

Noise (label ``-1``) is excluded from cluster metrics but kept in coverage.
Degenerate comparisons return ``None`` instead of a perfect score.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

__all__ = [
    "STABILITY_QUANTILES",
    "partition_metrics",
    "stability_pair",
    "summarize_stability",
    "within_dispersion",
]

STABILITY_QUANTILES: tuple[float, float] = (0.1, 0.9)
"""Descriptive lower/upper quantiles of the replica ARI distribution."""


def partition_metrics(
    labels: np.ndarray,
    reference_distances: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    """Describe a partition and score it on the common metric sample.

    Parameters
    ----------
    labels : numpy.ndarray of shape (n,)
        Labels of every training row; negative values denote noise.
    reference_distances : numpy.ndarray of shape (m, m)
        Reference distances between the rows selected by ``indices``.
    indices : numpy.ndarray of shape (m,)
        Positions of the metric sample within ``labels``.

    Returns
    -------
    dict
        Cluster count, coverage, noise, cluster sizes, reference
        silhouette (or ``None`` with a reason) and evaluated support.
    """
    labels = np.asarray(labels)
    assigned = labels >= 0
    _, counts = np.unique(labels[assigned], return_counts=True)
    local = labels[indices]
    keep = local >= 0
    subset = local[keep]
    n_local_clusters = len(np.unique(subset))
    score: float | None = None
    dispersion: float | None = None
    reason: str | None = None
    if not 2 <= n_local_clusters < len(subset):
        reason = (
            "silhouette requires 2 <= clusters < assigned metric observations"
        )
    else:
        distances = reference_distances[np.ix_(keep, keep)].copy()
        np.fill_diagonal(distances, 0)
        score = float(
            silhouette_score(distances, subset, metric="precomputed")
        )
        dispersion = within_dispersion(distances, subset)
    has_clusters = len(counts) > 0
    return {
        "n_clusters": len(counts),
        "coverage": float(assigned.mean()),
        "noise_count": int((~assigned).sum()),
        "min_cluster_size": int(np.min(counts)) if has_clusters else None,
        "max_cluster_size": int(np.max(counts)) if has_clusters else None,
        "silhouette": score,
        "silhouette_reason": reason,
        "metric_assigned_count": int(keep.sum()),
        "within_dispersion": dispersion,
    }


def within_dispersion(distances: np.ndarray, labels: np.ndarray) -> float:
    """Pooled within-cluster dispersion ``W`` of the gap statistic.

    ``W = sum_r D_r / (2 n_r)`` where ``D_r`` is the sum of the pairwise
    reference distances inside cluster ``r`` and ``n_r`` its size
    (Tibshirani, Walther and Hastie, 2001).

    Parameters
    ----------
    distances : numpy.ndarray of shape (m, m)
        Reference distances between assigned observations.
    labels : numpy.ndarray of shape (m,)
        Their cluster labels.
    """
    total = 0.0
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        block = distances[np.ix_(members, members)]
        total += float(block.sum()) / (2 * len(members))
    return total


def stability_pair(
    reference: np.ndarray, replica: np.ndarray, min_common: int
) -> dict[str, Any]:
    """Compare two partitions of the same rows on their common support.

    The required support is ``max(min_common, 2 * k)``, where ``k`` is the
    largest number of clusters represented in either partition.

    Parameters
    ----------
    reference, replica : numpy.ndarray of shape (n,)
        Labels of the same rows; negative values denote noise.
    min_common : int
        Floor of the common assigned support.

    Returns
    -------
    dict
        ARI (or ``None``), common support, required support, agreement of
        the assigned/noise status and the reason when not evaluable.
    """
    reference = np.asarray(reference)
    replica = np.asarray(replica)
    assigned = (reference >= 0) & (replica >= 0)
    support = int(assigned.sum())
    agreement = float(np.mean((reference >= 0) == (replica >= 0)))
    a, b = reference[assigned], replica[assigned]
    ka, kb = len(np.unique(a)), len(np.unique(b))
    required = max(min_common, 2 * max(ka, kb))
    valid = support >= required and 2 <= ka < support and 2 <= kb < support
    return {
        "ari": float(adjusted_rand_score(a, b)) if valid else None,
        "common_assigned": support,
        "required_support": required,
        "assignment_agreement": agreement,
        "reason": None if valid else (
            "insufficient support or degenerate partition"
        ),
    }


def summarize_stability(
    records: Sequence[dict[str, Any]], expected: int
) -> dict[str, Any]:
    """Summarise replica comparisons of one stability axis.

    Parameters
    ----------
    records : sequence of dict
        Replica comparisons; ``ari`` may be ``None``.
    expected : int
        Number of planned replicas (denominator of the evaluable fraction).
    """
    scores = [r["ari"] for r in records if r.get("ari") is not None]
    low, high = STABILITY_QUANTILES
    return {
        "median": float(np.median(scores)) if scores else None,
        "q10": float(np.quantile(scores, low)) if scores else None,
        "q90": float(np.quantile(scores, high)) if scores else None,
        "evaluable_fraction": len(scores) / expected,
        "completed": len(records),
        "expected": expected,
    }
