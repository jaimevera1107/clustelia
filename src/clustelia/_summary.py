"""Lightweight, dependency-free cluster-label diagnostics."""

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable


@dataclass(frozen=True)
class ClusterSummary:
    """Summary of assigned cluster labels.

    ``counts`` and ``shares`` include the noise label, when present.
    ``shares`` are relative to all observations, including noise.
    """

    n_samples: int
    n_clusters: int
    n_noise: int
    counts: dict[Hashable, int]
    shares: dict[Hashable, float]


def summarize_clusters(
    labels: Iterable[Hashable], *, noise_label: Hashable = -1
) -> ClusterSummary:
    """Summarize a sequence of cluster assignments.

    Parameters
    ----------
    labels:
        Hashable cluster labels. Iterators are consumed once. Empty input is
        accepted. Labels may be strings, integers, or other hashable values.
    noise_label:
        Label interpreted as unassigned/noise (default ``-1``, common in
        DBSCAN/HDBSCAN). Noise does not count toward ``n_clusters``.

    Returns
    -------
    ClusterSummary
        Counts, fractions and sample/cluster/noise totals. No fitted model
        or input data matrix is required.

    Examples
    --------
    >>> s = summarize_clusters([0, 0, 1, -1])
    >>> (s.n_samples, s.n_clusters, s.n_noise)
    (4, 2, 1)
    >>> s.shares[0]
    0.5
    """
    counts = dict(Counter(labels))
    n_samples = sum(counts.values())
    n_noise = counts.get(noise_label, 0)
    n_clusters = len(counts) - int(noise_label in counts)
    shares = {label: count / n_samples for label, count in counts.items()}
    return ClusterSummary(
        n_samples=n_samples,
        n_clusters=n_clusters,
        n_noise=n_noise,
        counts=counts,
        shares=shares,
    )
