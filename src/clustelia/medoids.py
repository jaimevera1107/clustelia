"""Alternating medoid updates on a precomputed distance matrix."""

from __future__ import annotations

import numpy as np

from ._validation import require_int
from .geometry import GeometryValues, ReferenceGeometry

__all__ = ["DEFAULT_RESTARTS", "AlternateMedoids"]

DEFAULT_RESTARTS: int = 5
"""Seeded restarts used by the ``medoids`` adapter (see VALIDATION)."""


class AlternateMedoids:
    """Alternating within-cluster medoid updates.

    This is a local descent heuristic, not PAM and not a global optimiser;
    different initial medoids may converge to partitions of different
    quality.

    Parameters
    ----------
    k : int
        Number of medoids.
    random_state : int
        Seed of the initial medoid draw.
    max_iter : int
        Maximum number of assignment/update rounds.
    n_init : int
        Number of seeded restarts; the run with the lowest total distance
        to its medoid is kept (ties keep the earliest). Restart 0 draws its
        medoids exactly as a single run does, so ``n_init=1`` reproduces
        earlier releases.
    """

    def __init__(
        self, k: int, random_state: int, max_iter: int = 100,
        n_init: int = 1,
    ) -> None:
        self.k = require_int("k", k, 1)
        self.random_state = random_state
        self.max_iter = require_int("max_iter", max_iter, 1)
        self.n_init = require_int("n_init", n_init, 1)

    def fit_predict(self, D: np.ndarray) -> np.ndarray:
        """Fit on a square distance matrix and return the labels.

        Parameters
        ----------
        D : numpy.ndarray of shape (n, n)
            Precomputed dissimilarities.

        Returns
        -------
        numpy.ndarray of shape (n,)
            Index of the closest medoid for every observation.

        Raises
        ------
        ValueError
            If ``D`` is not square or has fewer rows than ``k``.
        """
        D = np.asarray(D)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError("D must be a square distance matrix.")
        n = len(D)
        if n < self.k:
            raise ValueError("D must have at least k observations.")
        rng = np.random.default_rng(self.random_state)
        best: tuple[float, np.ndarray, bool, int] | None = None
        for _ in range(self.n_init):
            run = self._single_run(D, rng.choice(n, self.k, replace=False))
            if best is None or run[0] < best[0]:
                best = run
        assert best is not None  # n_init >= 1
        self.inertia_, medoids, self.converged_, self.n_iter_ = best
        self.medoid_indices_ = medoids
        self.labels_ = D[:, medoids].argmin(axis=1)
        return self.labels_

    def _single_run(
        self, D: np.ndarray, medoids: np.ndarray
    ) -> tuple[float, np.ndarray, bool, int]:
        """Alternate from ``medoids``; return cost, medoids, convergence."""
        converged = False
        iterations = 0
        for _ in range(self.max_iter):
            iterations += 1
            updated = self._update(D, medoids)
            if np.array_equal(updated, medoids):
                converged = True
                break
            medoids = updated
        labels = D[:, medoids].argmin(axis=1)
        cost = float(D[np.arange(len(D)), medoids[labels]].sum())
        return cost, medoids, converged, iterations

    def _update(self, D: np.ndarray, medoids: np.ndarray) -> np.ndarray:
        """Move each medoid to the member minimising within-cluster cost."""
        labels = D[:, medoids].argmin(axis=1)
        updated = medoids.copy()
        for cluster in range(self.k):
            members = np.flatnonzero(labels == cluster)
            if len(members):
                cost = D[np.ix_(members, members)].sum(axis=1)
                updated[cluster] = members[np.argmin(cost)]
        return updated


class SampledMedoids:
    """CLARA-style medoids with bounded sampled pairwise workspaces.

    Each trial optimises medoids on a subsample and evaluates its objective
    on the *full* training table by row blocks against the sampled medoids.
    This is a deterministic sampled heuristic, NOT an exact PAM solution.

    No full n-by-n distance matrix is allocated by this estimator; the
    experiment's separate reference validation still follows its own
    memory budget and common metric subsampling policies.
    """

    def __init__(
        self, k: int, random_state: int, *, sample_size: int = 64,
        samples: int = 5, max_iter: int = 100, max_pairwise_mb: float = 256,
        block_rows: int = 256,
    ) -> None:
        self.k = require_int("k", k, 2)
        self.seed = int(random_state)
        self.sample_size = require_int("sample_size", sample_size, 2)
        self.samples = require_int("samples", samples, 1)
        self.max_iter = require_int("max_iter", max_iter, 1)
        self.block_rows = require_int("block_rows", block_rows, 1)
        self.max_pairwise_mb = float(max_pairwise_mb)

    def fit_predict(
        self, values: GeometryValues, geometry: ReferenceGeometry
    ) -> np.ndarray:
        """Fit sampled medoids on transformed values and reference geometry."""
        from .geometry import guard_pairwise

        n = len(values)
        if n < self.k:
            raise ValueError("CLARA requires at least k observations")
        size = min(n, max(self.sample_size, 2 * self.k + 1))
        guard_pairwise(size, self.max_pairwise_mb)
        # Limit cross-distance workspace too, not just sampled s-by-s.
        max_block_rows = int(
            self.max_pairwise_mb * 1024**2 / (3 * 8 * self.k)
        )
        if max_block_rows < 1:
            raise ValueError("CLARA pairwise budget cannot hold one row block")
        rows_per_block = min(self.block_rows, max_block_rows)
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.samples):
            indices = np.sort(rng.choice(n, size=size, replace=False))
            sample = _select_rows(values, indices)
            distances = geometry.distances(sample)
            local_seed = int(rng.integers(0, 2**32 - 1))
            run = AlternateMedoids(self.k, local_seed, self.max_iter,
                                   n_init=3)
            run.fit_predict(distances)
            selected = indices[run.medoid_indices_]
            protos = _select_rows(values, selected)
            labels = np.empty(n, dtype=int)
            cost = 0.0
            for start, stop, distances in geometry.distance_blocks(
                values, protos, block_rows=rows_per_block
            ):
                nearest = distances.argmin(axis=1)
                labels[start:stop] = nearest
                cost += float(
                    distances[np.arange(stop - start), nearest].sum()
                )
            if best is None or cost < best[0]:
                best = (cost, selected.copy(), labels.copy(),
                        run.converged_, run.n_iter_)
        assert best is not None
        self.inertia_, self.medoid_indices_, self.labels_, self.converged_, \
            self.n_iter_ = best
        return self.labels_.copy()


def _select_rows(
    values: GeometryValues, indices: np.ndarray
) -> GeometryValues:
    """Select transformed rows with positional semantics."""
    if hasattr(values, "iloc"):
        return values.iloc[indices].copy()
    return np.asarray(values)[indices].copy()
