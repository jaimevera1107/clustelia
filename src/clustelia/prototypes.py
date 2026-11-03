"""Gower k-prototypes: centroid-style clustering for mixed data.

The estimator alternates two steps in the metric coding of a fitted
:class:`~clustelia.geometry.GowerMetric`:

1. assign each row to its closest prototype by weighted Gower distance;
2. replace each prototype, variable by variable, with the value that
   minimises the summed contributions of its members: the median for
   numeric variables, the mode for nominal and binary ones and the lower
   median position for ordinal ones.

Without asymmetric binary variables every pair compares every variable, so
the Gower denominator is constant and step 2 is an exact coordinate-wise
minimiser: the total cost never increases and the descent converges to a
local optimum. With asymmetric binaries the denominator depends on the
pair and step 2 becomes a heuristic. On nominal-only data the method is
exactly k-modes with simple matching (Huang, 1998). It is not Huang's
k-prototypes, whose numeric part is squared euclidean and whose balance
is a free ``gamma`` parameter; here the balance is the Gower weighting.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

import numpy as np
import pandas as pd

from ._validation import require_int
from .geometry import GowerMetric

__all__ = ["DEFAULT_RESTARTS", "GowerPrototypes"]

DEFAULT_RESTARTS: int = 5
"""Seeded restarts; the run with the lowest total cost is kept."""

_DEFAULT_MAX_ITER: int = 100


def _first_mode(values: np.ndarray) -> Any:
    """Most frequent value; ties go to the value seen first."""
    codes, uniques = pd.factorize(values, sort=False)
    return uniques[int(np.argmax(np.bincount(codes)))]


def _lower_median(values: np.ndarray) -> float:
    """Lower median, so the result is always an observed value."""
    ordered = np.sort(values.astype(float))
    return float(ordered[(len(ordered) - 1) // 2])


def _observed(
    values: pd.Series, summary: Callable[[np.ndarray], Any]
) -> Any:
    """Apply ``summary`` to the observed values; NaN when none observed.

    Complete columns take the original path unchanged.
    """
    if not values.isna().any():
        return summary(values.to_numpy())
    observed = values.dropna()
    return summary(observed.to_numpy()) if len(observed) else np.nan


class GowerPrototypes:
    """k-prototypes under a fitted weighted Gower metric.

    This is a local descent heuristic: different initialisations may reach
    partitions of different quality, which is why several seeded restarts
    are run and the lowest cost is kept.

    Parameters
    ----------
    k : int
        Number of prototypes.
    random_state : int
        Seed of the initialisation draws.
    n_init : int
        Number of restarts.
    max_iter : int
        Maximum assignment/update rounds per restart.

    Attributes
    ----------
    prototypes_ : pandas.DataFrame
        Prototypes in metric coding, one row per cluster.
    labels_ : numpy.ndarray
        Training labels.
    inertia_ : float
        Sum of the Gower distances of the rows to their prototype.
    converged_ : bool
        Whether the kept restart stopped before ``max_iter``.
    n_iter_ : int
        Rounds used by the kept restart.
    """

    def __init__(
        self,
        k: int,
        random_state: int,
        n_init: int = DEFAULT_RESTARTS,
        max_iter: int = _DEFAULT_MAX_ITER,
    ) -> None:
        self.k = require_int("k", k, 1)
        self.random_state = random_state
        self.n_init = require_int("n_init", n_init, 1)
        self.max_iter = require_int("max_iter", max_iter, 1)

    def get_params(self, deep: bool = False) -> dict[str, Any]:
        """Return the constructor parameters (scikit-learn convention)."""
        return {
            "k": self.k,
            "random_state": self.random_state,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
        }

    def fit(self, X: pd.DataFrame, metric: GowerMetric) -> Self:
        """Fit on rows already in the metric coding of ``metric``.

        Parameters
        ----------
        X : pandas.DataFrame
            Output of ``metric.transform`` for the training rows.
        metric : GowerMetric
            Fitted metric defining distances and variable kinds.

        Returns
        -------
        GowerPrototypes
            The fitted estimator.

        Raises
        ------
        TypeError
            If ``X`` is not a DataFrame.
        ValueError
            If ``X`` has fewer distinct rows than ``k``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("GowerPrototypes requires a DataFrame input.")
        if len(X.drop_duplicates()) < self.k:
            raise ValueError("X must have at least k distinct rows.")
        self.metric_ = metric
        rows = X.reset_index(drop=True)
        # Medians are real-valued even when a numeric column is integral.
        for column in metric.numeric:
            rows[column] = rows[column].to_numpy(dtype=float)
        rng = np.random.default_rng(self.random_state)
        best: tuple[float, pd.DataFrame, np.ndarray, bool, int] | None = None
        for _ in range(self.n_init):
            seed = int(rng.integers(0, 2**32))
            outcome = self._single_run(rows, seed)
            # Strict improvement keeps the earliest restart on ties.
            if best is None or outcome[0] < best[0]:
                best = outcome
        assert best is not None
        self.inertia_, self.prototypes_, self.labels_ = best[:3]
        self.converged_, self.n_iter_ = best[3], best[4]
        return self

    def fit_predict(self, X: pd.DataFrame, metric: GowerMetric) -> np.ndarray:
        """Fit and return the training labels."""
        return self.fit(X, metric).labels_.copy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Assign rows in metric coding to the closest prototype."""
        distances = self.metric_.distances(X, self.prototypes_)
        return np.asarray(distances.argmin(axis=1))

    def _single_run(
        self, rows: pd.DataFrame, seed: int
    ) -> tuple[float, pd.DataFrame, np.ndarray, bool, int]:
        """One initialisation followed by alternating updates."""
        prototypes = self._initial_prototypes(rows, seed)
        labels = np.full(len(rows), -1)
        converged = False
        iterations = 0
        for _ in range(self.max_iter):
            iterations += 1
            distances = self.metric_.distances(rows, prototypes)
            updated = distances.argmin(axis=1)
            updated, prototypes = self._repair_empty(
                rows, distances, updated, prototypes
            )
            if np.array_equal(updated, labels):
                converged = True
                break
            labels = updated
            prototypes = self._update(rows, labels, prototypes)
        distances = self.metric_.distances(rows, prototypes)
        labels = distances.argmin(axis=1)
        cost = float(distances[np.arange(len(rows)), labels].sum())
        return cost, prototypes, labels, converged, iterations

    def _initial_prototypes(
        self, rows: pd.DataFrame, seed: int
    ) -> pd.DataFrame:
        """Distance-weighted seeding (k-means++ style, Gower distances)."""
        rng = np.random.default_rng(seed)
        chosen = [int(rng.integers(len(rows)))]
        closest = self.metric_.distances(rows, rows.iloc[chosen]).ravel()
        while len(chosen) < self.k:
            total = closest.sum()
            if total > 0:
                candidate = int(rng.choice(len(rows), p=closest / total))
            else:
                # Every row coincides with a prototype; draw uniformly.
                candidate = int(rng.integers(len(rows)))
            if closest[candidate] == 0:
                remaining = np.flatnonzero(closest > 0)
                if not len(remaining):
                    raise ValueError("X must have at least k distinct rows.")
                candidate = int(rng.choice(remaining))
            chosen.append(candidate)
            new = self.metric_.distances(rows, rows.iloc[[candidate]])
            closest = np.minimum(closest, new.ravel())
        return rows.iloc[chosen].reset_index(drop=True)

    def _repair_empty(
        self,
        rows: pd.DataFrame,
        distances: np.ndarray,
        labels: np.ndarray,
        prototypes: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Reseed empty clusters with the worst-served rows.

        Each empty cluster takes the row farthest from its current
        prototype (deterministic argmax), which then forms a singleton.
        """
        labels = labels.copy()
        prototypes = prototypes.copy()
        served = distances[np.arange(len(rows)), labels].copy()
        for cluster in range(self.k):
            if (labels == cluster).any():
                continue
            donor = int(np.argmax(served))
            labels[donor] = cluster
            served[donor] = -np.inf
            prototypes.iloc[cluster] = rows.iloc[donor]
        return labels, prototypes

    def _update(
        self,
        rows: pd.DataFrame,
        labels: np.ndarray,
        prototypes: pd.DataFrame,
    ) -> pd.DataFrame:
        """Coordinate-wise prototype update for every non-empty cluster."""
        metric = self.metric_
        updated = prototypes.copy()
        for cluster in range(self.k):
            members = rows.iloc[np.flatnonzero(labels == cluster)]
            if members.empty:
                continue
            for column in metric.numeric:
                values = members[column].to_numpy(dtype=float)
                if np.isnan(values).any():
                    # Pairwise deletion: summarise observed values only;
                    # a column unobserved in the cluster stays missing.
                    values = values[~np.isnan(values)]
                    updated.at[cluster, column] = (
                        float(np.median(values)) if len(values) else np.nan
                    )
                else:
                    updated.at[cluster, column] = float(np.median(values))
            for column in metric.ordinal:
                updated.at[cluster, column] = _observed(
                    members[column], _lower_median
                )
            for column in metric.categorical + metric.binary:
                updated.at[cluster, column] = _observed(
                    members[column], _first_mode
                )
        return updated
