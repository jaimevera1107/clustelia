"""Null-reference test: is the partition better than structureless data?

Stability alone cannot tell clusters from noise: k-means splits a uniform
square or a single Gaussian into stable, reproducible pieces. Before a
compact-objective candidate is recommended, the explored candidates are
refitted on reference datasets without cluster structure. The best
reference silhouette reached on the data must exceed, in every replicate,
the best silhouette the same candidates reach on the null dataset. Using
the best value on both sides mirrors the selection over algorithms and
``k``; comparing only the winning candidate is optimistic on
structureless data (measured: about one uniform square in three passed).

Reference families
------------------
- ``pca_uniform`` (default for Euclidean geometry): uniform in the box
  spanned by the training data along its principal axes (reference (b)
  of the gap statistic, Tibshirani, Walther and Hastie, 2001). It is
  sensitive to real groups but reads skewed unimodal marginals as
  structure: in development studies a lognormal single mode without
  groups was recommended in roughly 7 of 10 runs.
- ``gaussian_copula`` (experimental, opt in): a Gaussian copula fitted
  on the normal scores of the data, mapped back through the empirical
  marginals. Every marginal distribution and the rank correlation are
  preserved exactly in distribution, so skewness alone is not read as
  structure; the price is a marked loss of sensitivity on unbalanced
  mixtures (3 of 30 detections in the same studies). Numeric features
  only.
- ``permutation`` (default for Gower geometry): each feature column
  permuted independently, which keeps every marginal distribution and
  removes the association between variables that clusters imply.

The family is chosen by :attr:`SelectionPolicy.null_family`; ``"auto"``
applies the geometry default above.

Decision rule
-------------
With ``B`` replicates the search passes only if its statistic is strictly
greater than the null maximum of every replicate. When the data follow
the reference, observed and null statistics are exchangeable (with the
default ``null_scope="grid"``), a rank test has a conditional Monte
Carlo resolution of ``1 / (B + 1)`` (0.1 for ``B = 9``). The fitted PCA
box and column-permutation references are data-dependent, so exchangeability
and a general false-recommendation bound are NOT guaranteed. Only one
screen is run per search. Failed screens abstain without retries; failed
backend fits instead yield an incomplete experiment. Replicates stop at
the first null maximum reaching the observed value.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtri

from ._seeding import candidate_id
from .config import (
    CandidateSpec,
    GeometrySpec,
    Schema,
    SelectionPolicy,
    candidate_parameters,
)
from .evaluation import partition_metrics
from .execution import (
    BACKEND_ERRORS,
    FitBudgetExhaustedError,
    FitExecutor,
)
from .geometry import ReferenceGeometry, guard_pairwise

__all__ = [
    "NULL_FAMILIES",
    "StructureOutcome",
    "StructureTester",
    "NULL_SCOPES",
    "family_grid",
    "family_key",
    "null_grid",
    "generate_null",
    "null_family",
    "structure_reserve",
]

NULL_FAMILIES: dict[str, str] = {
    "euclidean": "pca_uniform",
    "gower": "permutation",
}
"""Default reference family of each geometry kind (``null_family="auto"``).
"""

_WORST_SILHOUETTE: float = -1.0
"""Value given to a null partition without an evaluable silhouette."""


def null_family(geometry_kind: str, requested: str = "auto") -> str:
    """Resolve the reference family for a geometry kind.

    Parameters
    ----------
    geometry_kind : {"euclidean", "gower"}
        Kind of the reference geometry.
    requested : str
        :attr:`SelectionPolicy.null_family`; ``"auto"`` returns the
        geometry default from :data:`NULL_FAMILIES`.

    Raises
    ------
    ValueError
        If a numeric-only family is requested for Gower geometry, which
        may carry non-numeric features.
    """
    if requested == "auto":
        return NULL_FAMILIES[geometry_kind]
    if geometry_kind == "gower" and requested != "permutation":
        raise ValueError(
            f"null_family={requested!r} needs numeric features; Gower "
            "geometry supports only 'permutation' (or 'auto')."
        )
    return requested


def family_key(spec: CandidateSpec) -> str:
    """Stable identifier of a candidate's family (all parameters but k)."""
    parameters = candidate_parameters(spec)
    parameters.pop("k")
    payload = json.dumps(parameters, sort_keys=True).encode()
    return "family-" + hashlib.sha256(payload).hexdigest()[:16]


def family_grid(
    spec: CandidateSpec, specs: Sequence[CandidateSpec]
) -> list[CandidateSpec]:
    """Members of ``specs`` in the same family as ``spec``, in order."""
    key = family_key(spec)
    return [s for s in specs if family_key(s) == key]


NULL_SCOPES: tuple[str, ...] = ("grid", "family")
"""Candidates refitted on each null dataset: the whole explored grid, or
only the preferred candidate's family."""


def null_grid(
    spec: CandidateSpec, specs: Sequence[CandidateSpec], scope: str
) -> list[CandidateSpec]:
    """Candidates whose best silhouette defines the test statistic.

    Raises
    ------
    ValueError
        If ``scope`` is unknown.
    """
    if scope == "grid":
        return list(specs) or [spec]
    if scope == "family":
        return family_grid(spec, specs) or [spec]
    raise ValueError(f"null_scope must be one of {NULL_SCOPES}.")


def structure_reserve(
    specs: Sequence[CandidateSpec], policy: SelectionPolicy, objective: str
) -> int:
    """Fits set aside to test one recommendation against the reference.

    The reserve covers the null grid of the worst case: every candidate
    for ``scope="grid"``, the largest family for ``scope="family"``. It is
    0 when the test does not apply (non-compact objectives) or is disabled
    (``null_replicates == 0``).
    """
    if objective != "compact" or policy.null_replicates == 0 or not specs:
        return 0
    largest = max(
        len(null_grid(s, specs, policy.null_scope)) for s in specs
    )
    return policy.null_replicates * largest


def _permutation_null(
    frame: pd.DataFrame,
    schema: Schema,
    varying: Sequence[str],
    rng: np.random.Generator,
    blocks: tuple[tuple, ...],
) -> pd.DataFrame:
    """Permute every varying feature column independently (or by block)."""
    null = frame.copy()
    seen: set[Any] = set()
    for block in blocks:
        if any(col not in schema.features for col in block):
            raise ValueError("Null permutation blocks must name features")
        if any(col in seen for col in block):
            raise ValueError("Null permutation blocks cannot overlap")
        seen.update(block)
        cols = [col for col in block if col in varying]
        if cols:
            permutation = rng.permutation(len(frame))
            null[cols] = frame[cols].iloc[permutation].to_numpy()
    for column in varying:
        if column not in seen:
            null[column] = frame[column].to_numpy()[
                rng.permutation(len(frame))
            ]
    return null


def _numeric_values(
    family: str, frame: pd.DataFrame, schema: Schema, varying: Sequence[str]
) -> np.ndarray:
    """Float matrix of the varying features, or raise for non-numeric."""
    if any(c not in schema.numeric for c in varying):
        raise ValueError(f"The {family} reference needs numeric features.")
    return frame[list(varying)].to_numpy(dtype=float)


def _pca_uniform_null(
    frame: pd.DataFrame,
    schema: Schema,
    varying: Sequence[str],
    rng: np.random.Generator,
    blocks: tuple[tuple, ...],
) -> pd.DataFrame:
    """Uniform box aligned with the principal axes of the data."""
    null = frame.copy()
    values = _numeric_values("pca_uniform", frame, schema, varying)
    centre = values.mean(axis=0)
    _, _, axes = np.linalg.svd(values - centre, full_matrices=False)
    scores = (values - centre) @ axes.T
    box = rng.uniform(scores.min(axis=0), scores.max(axis=0),
                      size=scores.shape)
    drawn = box @ axes + centre
    for position, column in enumerate(varying):
        null[column] = drawn[:, position]
    return null


def _gaussian_copula_null(
    frame: pd.DataFrame,
    schema: Schema,
    varying: Sequence[str],
    rng: np.random.Generator,
    blocks: tuple[tuple, ...],
) -> pd.DataFrame:
    """Gaussian copula on the normal scores, mapped back by rank.

    Each column is converted to normal scores through its mid-ranks, a
    multivariate normal with the correlation of those scores is drawn,
    and every drawn column is mapped back to the observed values of the
    same column by rank. The null therefore holds exactly the observed
    values of each feature (a permutation per column, as the permutation
    family does) while reproducing the linear dependence between
    features instead of removing it.
    """
    null = frame.copy()
    values = _numeric_values("gaussian_copula", frame, schema, varying)
    n_rows, n_cols = values.shape
    # Mid-ranks keep ties symmetric; the (r - 0.5) / n offset avoids the
    # infinite quantiles at 0 and 1.
    ranks = pd.DataFrame(values).rank(method="average").to_numpy()
    scores = _normal_quantile((ranks - 0.5) / n_rows)
    correlation = np.corrcoef(scores, rowvar=False).reshape(n_cols, n_cols)
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)
    # A small ridge keeps the factorisation valid for nearly collinear
    # scores; it perturbs the correlation at the fourth decimal at most.
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.clip(eigenvalues, _COPULA_RIDGE, None)
    root = eigenvectors * np.sqrt(eigenvalues)
    drawn = rng.standard_normal(size=(n_rows, n_cols)) @ root.T
    for position, column in enumerate(varying):
        sorted_values = np.sort(values[:, position])
        order = np.argsort(drawn[:, position], kind="stable")
        back = np.empty(n_rows, dtype=float)
        back[order] = sorted_values
        null[column] = back
    return null


def _normal_quantile(probabilities: np.ndarray) -> np.ndarray:
    """Standard normal quantile function (vectorised)."""
    return np.asarray(ndtri(probabilities), dtype=float)


_COPULA_RIDGE: float = 1e-6
"""Smallest eigenvalue kept in the copula correlation factorisation."""

_NullGenerator = Callable[
    [pd.DataFrame, Schema, Sequence[str], np.random.Generator,
     tuple[tuple, ...]],
    pd.DataFrame,
]

_GENERATORS: dict[str, _NullGenerator] = {
    "permutation": _permutation_null,
    "pca_uniform": _pca_uniform_null,
    "gaussian_copula": _gaussian_copula_null,
}
"""Reference generator of every supported family."""


def generate_null(
    family: str,
    frame: pd.DataFrame,
    schema: Schema,
    rng: np.random.Generator,
    *,
    blocks: tuple[tuple, ...] = (),
) -> pd.DataFrame:
    """Draw one structureless dataset with the shape of ``frame``.

    Only nonconstant feature columns are regenerated; constant features
    and identifier columns are copied unchanged so that the null pipeline
    sees the same column roles.

    Parameters
    ----------
    family : {"pca_uniform", "gaussian_copula", "permutation"}
        Reference family (see the module docstring).
    frame : pandas.DataFrame
        Validated training table.
    schema : Schema
        Resolved column roles.
    rng : numpy.random.Generator
        Source of randomness (seeded by the caller).
    blocks : tuple of tuple
        Disjoint groups shuffled with the SAME row permutation; preserves
        dependencies internal to each block. Used by ``permutation`` only.

    Returns
    -------
    pandas.DataFrame
        New table with the same index and columns.

    Raises
    ------
    ValueError
        If ``family`` is unknown, or a numeric family is used with
        non-numeric features.
    """
    generator = _GENERATORS.get(family)
    if generator is None:
        raise ValueError(
            f"Unknown null reference family: {family!r}; expected one of "
            f"{tuple(_GENERATORS)}."
        )
    varying = [c for c in schema.features if frame[c].nunique() > 1]
    return generator(frame, schema, varying, rng, blocks)


class _BudgetExhaustedError(Exception):
    """Internal signal: the budget ran out inside a null replicate."""


def _null_metrics(
    labels: np.ndarray, distances: np.ndarray, indices: np.ndarray
) -> tuple[float | None, float | None]:
    """Silhouette and within dispersion of one null partition.

    Module-level so that it can run inside worker processes.
    """
    metrics = partition_metrics(labels, distances, indices)
    return metrics["silhouette"], metrics["within_dispersion"]


@dataclass
class StructureOutcome:
    """Result of testing a search against the null reference.

    Attributes
    ----------
    observed : float
        Best reference silhouette of the grid on the training data.
    family : str
        Reference family used.
    grid : list of str
        Identifiers of the candidates refitted on each null dataset.
    passed : bool
        Whether the observed value exceeded the null maximum of every
        replicate.
    exhausted : bool
        Whether the fit or time budget stopped the test before a decision.
    null_maxima : list of float
        Best grid silhouette of each replicate run.
    null_silhouettes : dict
        Silhouette of every grid candidate in each replicate run (the
        worst value -1 when a null partition has no silhouette).
    reason : str or None
        Why the search did not pass, if it did not.
    """

    observed: float
    family: str
    grid: list[str]
    passed: bool = False
    exhausted: bool = False
    failed: bool = False
    null_maxima: list[float] = field(default_factory=list)
    null_silhouettes: dict[str, list[float]] = field(default_factory=dict)
    null_dispersions: dict[str, list[float | None]] = field(
        default_factory=dict
    )
    reason: str | None = None

    def as_record(self) -> dict[str, Any]:
        """Plain-dictionary form stored in the candidate record."""
        return {
            "observed": self.observed,
            "family": self.family,
            "grid": list(self.grid),
            "passed": self.passed,
            "exhausted": self.exhausted,
            "failed": self.failed,
            "null_maxima": list(self.null_maxima),
            "replicates": len(self.null_maxima),
            "reason": self.reason,
        }


class StructureTester:
    """Refit the null grid on reference datasets under the shared budget.

    Parameters
    ----------
    executor : FitExecutor
        Executor charging every null fit to the budget.
    frame : pandas.DataFrame
        Validated training table.
    schema : Schema
        Resolved column roles.
    geometry : GeometrySpec
        Reference geometry, refitted on each null dataset.
    metric_indices : numpy.ndarray
        Row positions of the common metric sample (reused for nulls).
    policy : SelectionPolicy
        Provides ``null_replicates``.
    max_pairwise_mb : float
        Preventive pairwise memory budget.
    error_policy : {"record", "raise"}
        Whether a failing null fit is recorded or raised.
    """

    def __init__(
        self,
        executor: FitExecutor,
        frame: pd.DataFrame,
        schema: Schema,
        geometry: GeometrySpec,
        metric_indices: np.ndarray,
        policy: SelectionPolicy,
        max_pairwise_mb: float,
        error_policy: str,
    ) -> None:
        self.executor = executor
        self.frame = frame
        self.schema = schema
        self.geometry = geometry
        self.metric_indices = metric_indices
        self.policy = policy
        self.max_pairwise_mb = max_pairwise_mb
        self.error_policy = error_policy
        self.family = null_family(geometry.kind, policy.null_family)

    def test(
        self, observed: float, grid: Sequence[CandidateSpec]
    ) -> StructureOutcome:
        """Compare ``observed`` with the null maxima of ``grid``.

        Parameters
        ----------
        observed : float
            Best reference silhouette of ``grid`` on the training data.
        grid : sequence of CandidateSpec
            Candidates refitted on every null dataset (see
            :func:`null_grid`).
        """
        grid = list(grid)
        if not grid:
            raise ValueError("The null grid needs at least one candidate.")
        outcome = StructureOutcome(
            observed=float(observed),
            family=self.family,
            grid=[candidate_id(s) for s in grid],
            null_silhouettes={candidate_id(s): [] for s in grid},
            null_dispersions={candidate_id(s): [] for s in grid},
        )
        key = "grid-" + "-".join(sorted(outcome.grid))
        for replicate in range(self.policy.null_replicates):
            try:
                scores = self._null_scores(grid, key, replicate)
            except (_BudgetExhaustedError, FitBudgetExhaustedError):
                outcome.exhausted = True
                outcome.reason = "null reference test not completed"
                return outcome
            except BACKEND_ERRORS as exc:
                if self.error_policy == "raise":
                    raise
                outcome.failed = True
                outcome.reason = (
                    f"null reference fit failed: {type(exc).__name__}: "
                    f"{exc}"
                )
                return outcome
            for identifier, (score, dispersion) in scores.items():
                outcome.null_silhouettes[identifier].append(score)
                outcome.null_dispersions[identifier].append(dispersion)
            maximum = max(score for score, _ in scores.values())
            outcome.null_maxima.append(maximum)
            if maximum >= outcome.observed:
                outcome.reason = (
                    f"silhouette not above the {self.family} null reference"
                )
                return outcome
        outcome.passed = True
        return outcome

    def _null_scores(
        self, grid: Sequence[CandidateSpec], key: str, replicate: int
    ) -> dict[str, tuple[float, float | None]]:
        """Silhouette and dispersion of every grid candidate on one null."""
        if not self.executor.available():
            raise _BudgetExhaustedError
        phase = f"null-{replicate}"
        rng = np.random.default_rng(
            self.executor.seed(key, f"{phase}-data")
        )
        null = generate_null(
            self.family, self.frame, self.schema, rng,
            blocks=self.geometry.null_blocks,
        )
        reference = ReferenceGeometry(self.geometry, self.schema).fit(null)
        guard_pairwise(len(self.metric_indices), self.max_pairwise_mb)
        values = reference.transform(null.iloc[self.metric_indices])
        distances = reference.distances(values)
        measured = self.executor.fit_apply(
            grid, null, phase, "structure",
            partial(_null_metrics, distances=distances,
                    indices=self.metric_indices),
        )
        return {
            candidate_id(member): (
                _WORST_SILHOUETTE if score is None else float(score),
                dispersion,
            )
            for member, (score, dispersion) in zip(
                grid, measured, strict=True
            )
        }
