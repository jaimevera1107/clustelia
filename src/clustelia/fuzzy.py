"""Fuzzy C-Means adapter with explicit, locally seeded memberships."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances

from ._validation import require_int, require_positive
from .optional import require_optional

__all__ = ["FuzzyCMeans"]

DEFAULT_MAX_ITER: int = 300
DEFAULT_TOL: float = 1e-5


class FuzzyCMeans:
    """Fuzzy C-Means delegated to scikit-fuzzy.

    The initial membership matrix is drawn from a local generator and
    passed through ``init``; NumPy's global random state is never touched.
    Memberships are not calibrated probabilities.

    Parameters
    ----------
    k : int
        Number of clusters.
    fuzzifier : float
        Exponent ``m`` (> 1), fixed within an objective.
    seed : int
        Seed of the initial memberships.
    max_iter : int
        Maximum iterations of the backend.
    tol : float
        Stopping tolerance of the backend.
    """

    def __init__(
        self,
        k: int,
        fuzzifier: float,
        seed: int,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOL,
    ) -> None:
        self.k = require_int("k", k, 2)
        self.fuzzifier = fuzzifier
        self.seed = seed
        self.max_iter = require_int("max_iter", max_iter, 2)
        self.tol = require_positive("tol", tol)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit on ``X`` and return the hard (argmax) labels."""
        backend = require_optional("fuzzy")
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        initial = rng.dirichlet(np.ones(self.k), size=len(X)).T
        centers, u, _, _, history, iterations, fpc = backend.cluster.cmeans(
            X.T, self.k, self.fuzzifier, self.tol, self.max_iter, init=initial
        )
        self.cluster_centers_ = centers
        self.memberships_ = u.T.copy()
        self.objective_history_ = history.copy()
        self.n_iter_ = int(iterations)
        # The backend stops at max_iter - 1 rounds; reaching it is treated
        # conservatively as non-convergence.
        self.converged_ = bool(iterations < self.max_iter - 1)
        self.fpc_ = float(fpc)
        tiny = np.finfo(float).tiny
        log_u = np.log(np.maximum(u, tiny))
        self.entropy_ = float(-np.sum(u * log_u) / len(X))
        self._compute_xie_beni(X)
        return np.asarray(u.argmax(axis=0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new rows with the fitted centres kept fixed."""
        backend = require_optional("fuzzy")
        X = np.asarray(X, dtype=float)
        initial = np.full((self.k, len(X)), 1 / self.k)
        u, *_ = backend.cluster.cmeans_predict(
            X.T,
            self.cluster_centers_,
            self.fuzzifier,
            self.tol,
            self.max_iter,
            init=initial,
        )
        return np.asarray(u.argmax(axis=0))

    def _compute_xie_beni(self, X: np.ndarray) -> None:
        """Compute the FCM objective and the generalised XB_m index.

        ``m`` is fixed within an objective and never tuned by this score.
        Coincident centres yield ``None`` instead of an artificial value.
        """
        squared = pairwise_distances(
            X, self.cluster_centers_, metric="sqeuclidean"
        )
        between = pairwise_distances(
            self.cluster_centers_, metric="sqeuclidean"
        )
        np.fill_diagonal(between, np.inf)
        separation = float(between.min())
        weights = self.memberships_**self.fuzzifier
        self.objective_ = float(np.sum(weights * squared))
        if separation > np.finfo(float).eps:
            self.xie_beni_: float | None = self.objective_ / (
                len(X) * separation
            )
        else:
            self.xie_beni_ = None
