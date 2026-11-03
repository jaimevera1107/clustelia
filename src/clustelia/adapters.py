"""Adapter registry and the fitted candidate pipeline.

The registry validates capabilities (objective, geometry, assignment,
stochasticity) instead of only mapping names to classes. Each fitted
candidate owns its own reference geometry and optional PCA; fitted
transformations are never shared between candidates or replicas.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    OPTICS,
    AgglomerativeClustering,
    Birch,
    BisectingKMeans,
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from . import medoids as _medoid_module
from .config import CandidateSpec, GeometrySpec, Schema
from .exceptions import CapabilityError
from .fuzzy import FuzzyCMeans
from .geometry import (
    GeometryValues,
    GowerMetric,
    ReferenceGeometry,
    guard_pairwise,
)
from .medoids import AlternateMedoids, SampledMedoids
from .optional import optional_version
from .prototypes import GowerPrototypes

__all__ = [
    "PAIRWISE_GUARDED",
    "REGISTRY",
    "AdapterCapability",
    "FittedCandidate",
    "Membership",
    "adapter_capabilities",
]


@dataclass(frozen=True)
class AdapterCapability:
    """Static capabilities of one adapter.

    Attributes
    ----------
    objective : str
        Objective whose primary metric can rank this adapter.
    geometries : tuple of str
        Reference geometries the adapter can fit in.
    assign : bool
        Whether new rows can be assigned without refitting.
    stochastic : bool
        Whether the result depends on the seed (adds a seed axis).
    backend : str
        Implementation providing the algorithm.
    optional_feature : str or None
        Optional extra required to fit, if any.
    notes : mapping
        Convention notes published by :func:`adapter_capabilities`.
    """

    objective: str
    geometries: tuple[str, ...]
    assign: bool
    stochastic: bool
    backend: str = "sklearn"
    optional_feature: str | None = None
    notes: Mapping[str, str] = field(default_factory=dict)


_EUCLIDEAN = ("euclidean",)
_GOWER = ("gower",)
_ANY = ("euclidean", "gower")

REGISTRY: Mapping[str, AdapterCapability] = {
    "kmeans": AdapterCapability("compact", _EUCLIDEAN, True, True),
    "medoids": AdapterCapability(
        "compact", _ANY, True, True, backend="clustelia.AlternateMedoids"
    ),
    "clara": AdapterCapability(
        "compact", _ANY, True, True, backend="clustelia.SampledMedoids",
        notes={"definition": "Sampled medoids, full-data blockwise scoring"},
    ),
    "agglomerative": AdapterCapability("compact", _EUCLIDEAN, False, False),
    "gmm": AdapterCapability("probabilistic", _EUCLIDEAN, True, True),
    "dbscan": AdapterCapability("density", _ANY, False, False),
    "minibatch": AdapterCapability("compact", _EUCLIDEAN, True, True),
    "birch": AdapterCapability("compact", _EUCLIDEAN, True, False),
    "bisecting": AdapterCapability("compact", _EUCLIDEAN, True, True),
    "spectral": AdapterCapability("compact", _EUCLIDEAN, False, True),
    "optics": AdapterCapability("density", _ANY, False, False),
    "hdbscan": AdapterCapability(
        "density",
        _ANY,
        False,
        False,
        notes={
            "min_samples_convention": (
                "includes the observation itself (sklearn backend)"
            )
        },
    ),
    "fcm": AdapterCapability(
        "fuzzy",
        _EUCLIDEAN,
        True,
        True,
        backend="scikit-fuzzy.cluster.cmeans",
        optional_feature="fuzzy",
    ),
    "bgmm": AdapterCapability(
        "probabilistic",
        _EUCLIDEAN,
        True,
        True,
        notes={"selection_metric": "heldout_loglikelihood only; no BIC"},
    ),
    "kprototypes": AdapterCapability(
        "compact",
        _GOWER,
        True,
        True,
        backend="clustelia.GowerPrototypes",
        notes={
            "definition": (
                "Gower prototypes (median / mode / lower-median level); "
                "equals k-modes with simple matching on nominal-only data"
            )
        },
    ),
}

PAIRWISE_GUARDED: frozenset[str] = frozenset(
    {"agglomerative", "spectral", "dbscan", "optics", "hdbscan", "medoids"}
)
"""Adapters that may allocate dense ``n x n`` workspaces."""

_MIXTURES: frozenset[str] = frozenset({"gmm", "bgmm"})
_PROTOTYPE_ADAPTERS: frozenset[str] = frozenset({"kprototypes"})
_ITERATION_CAPPED: frozenset[str] = frozenset({"kmeans", "minibatch"})


def adapter_capabilities() -> dict[str, dict[str, Any]]:
    """Return a copy of the adapter capabilities and their availability.

    Returns
    -------
    dict
        One record per adapter with ``objective``, ``geometries``,
        ``assign``, ``stochastic``, ``backend``, ``available``,
        ``sample_weights``, ``sparse`` and any convention notes.
    """
    result: dict[str, dict[str, Any]] = {}
    for name, capability in REGISTRY.items():
        feature = capability.optional_feature
        record: dict[str, Any] = {
            "objective": capability.objective,
            "geometries": capability.geometries,
            "assign": capability.assign,
            "stochastic": capability.stochastic,
            "backend": capability.backend,
            "available": feature is None
            or optional_version(feature) is not None,
            "sample_weights": False,
            "sparse": False,
        }
        record.update(deepcopy(dict(capability.notes)))
        result[name] = record
    return result


@dataclass(frozen=True)
class Membership:
    """Soft assignment with explicit semantics.

    Attributes
    ----------
    semantics : str
        ``"mixture_posterior"``, ``"fuzzy_membership"`` or
        ``"density_association_strength"``.
    values : numpy.ndarray
        Membership matrix, one row per observation.
    columns : tuple
        Column labels of ``values``.
    normalization : str
        Whether rows form a distribution over clusters.
    """

    semantics: str
    values: np.ndarray
    columns: tuple
    normalization: str = "rows_sum_to_one"


EstimatorFactory = Callable[[CandidateSpec, int, str], Any]


def _kmeans(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return KMeans(
        n_clusters=spec.k, n_init=10, max_iter=300, tol=1e-4,
        algorithm="lloyd", random_state=seed,
    )


def _agglomerative(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return AgglomerativeClustering(
        n_clusters=spec.k, metric="euclidean", linkage="ward",
        compute_distances=True,
    )


def _gmm(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return GaussianMixture(
        n_components=spec.k, covariance_type=spec.covariance_type, n_init=3,
        max_iter=300, reg_covar=1e-6, tol=1e-3, random_state=seed,
    )


def _bgmm(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return BayesianGaussianMixture(
        n_components=spec.k, covariance_type=spec.covariance_type, n_init=3,
        max_iter=300, reg_covar=1e-6, tol=1e-3, random_state=seed,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=spec.concentration_prior,
    )


def _minibatch(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return MiniBatchKMeans(
        n_clusters=spec.k, batch_size=spec.batch_size, n_init=10,
        max_iter=100, tol=0.0, max_no_improvement=10,
        reassignment_ratio=0.01, random_state=seed,
    )


def _birch(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return Birch(
        n_clusters=spec.k, threshold=spec.threshold, branching_factor=50,
        compute_labels=True,
    )


def _bisecting(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return BisectingKMeans(
        n_clusters=spec.k, init="k-means++", n_init=3, max_iter=300,
        tol=1e-4, algorithm="lloyd", bisecting_strategy="biggest_inertia",
        random_state=seed,
    )


def _spectral(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return SpectralClustering(
        n_clusters=spec.k, eigen_solver="arpack", n_init=10,
        gamma=spec.gamma, affinity="rbf", eigen_tol=0.0,
        assign_labels="kmeans", n_jobs=1, random_state=seed,
    )


def _fcm(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return FuzzyCMeans(spec.k or 0, spec.fuzzifier, seed)


def _dbscan(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return DBSCAN(
        eps=spec.eps, min_samples=spec.min_samples, metric=metric, n_jobs=1
    )


def _optics(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return OPTICS(
        min_samples=spec.min_samples, min_cluster_size=spec.min_cluster_size,
        max_eps=np.inf, metric=metric, cluster_method="xi", xi=spec.xi,
        predecessor_correction=True, algorithm="brute", n_jobs=1,
    )


def _hdbscan(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return HDBSCAN(
        min_samples=spec.min_samples, min_cluster_size=spec.min_cluster_size,
        metric=metric, cluster_selection_epsilon=0.0,
        cluster_selection_method="eom", alpha=1.0,
        allow_single_cluster=False, algorithm="auto", n_jobs=1, copy=True,
    )


def _medoids(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return AlternateMedoids(
        spec.k or 0, seed, n_init=_medoid_module.DEFAULT_RESTARTS
    )


def _clara(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return SampledMedoids(
        spec.k or 0, seed, sample_size=spec.clara_sample_size,
        samples=spec.clara_samples,
    )


def _kprototypes(spec: CandidateSpec, seed: int, metric: str) -> Any:
    return GowerPrototypes(spec.k or 0, seed)


_FACTORIES: Mapping[str, EstimatorFactory] = {
    "kprototypes": _kprototypes,
    "kmeans": _kmeans,
    "agglomerative": _agglomerative,
    "gmm": _gmm,
    "bgmm": _bgmm,
    "minibatch": _minibatch,
    "birch": _birch,
    "bisecting": _bisecting,
    "spectral": _spectral,
    "fcm": _fcm,
    "dbscan": _dbscan,
    "optics": _optics,
    "hdbscan": _hdbscan,
    "medoids": _medoids,
    "clara": _clara,
}


def _take_rows(values: GeometryValues, rows: np.ndarray) -> GeometryValues:
    """Select rows by position from an array or a DataFrame (copy)."""
    if isinstance(values, pd.DataFrame):
        return values.iloc[rows].copy()
    return np.asarray(values)[rows].copy()


class FittedCandidate:
    """Complete pipeline of one candidate: geometry, PCA and estimator.

    Parameters
    ----------
    spec : CandidateSpec
        Candidate configuration.
    geometry : GeometrySpec
        Reference geometry fitted inside the pipeline.
    schema : Schema
        Resolved column roles.
    seed : int
        Seed of every stochastic component.
    max_pairwise_mb : float
        Preventive pairwise memory budget.
    """

    def __init__(
        self,
        spec: CandidateSpec,
        geometry: GeometrySpec,
        schema: Schema,
        seed: int,
        max_pairwise_mb: float,
    ) -> None:
        self.spec = spec
        self.geometry_spec = geometry
        self.schema = schema
        self.seed = seed
        self.max_pairwise_mb = max_pairwise_mb

    @property
    def capability(self) -> AdapterCapability:
        """Capabilities of this candidate's adapter."""
        return REGISTRY[self.spec.algorithm]

    def fit(self, frame: pd.DataFrame) -> Self:
        """Fit the whole pipeline on ``frame`` and record diagnostics."""
        algorithm = self.spec.algorithm
        self.geometry_ = ReferenceGeometry(
            self.geometry_spec, self.schema
        ).fit(frame)
        self.reducer_: PCA | None = None
        values = self.geometry_.transform(frame)
        if self.spec.pca_variance is not None:
            self.reducer_ = PCA(
                n_components=self.spec.pca_variance, svd_solver="full"
            )
            values = self.reducer_.fit_transform(values)
        if algorithm in PAIRWISE_GUARDED:
            guard_pairwise(len(frame), self.max_pairwise_mb)
        metric = (
            "precomputed" if self.geometry_spec.kind == "gower"
            else "euclidean"
        )
        self.estimator_ = _FACTORIES[algorithm](self.spec, self.seed, metric)
        if algorithm == "clara":
            self.estimator_.max_pairwise_mb = self.max_pairwise_mb
        self.labels_ = self._fit_labels(values, metric)
        self.prototypes_: GeometryValues | None = None
        if algorithm in ("medoids", "clara"):
            self.prototypes_ = _take_rows(
                values, self.estimator_.medoid_indices_
            )
        elif algorithm in _PROTOTYPE_ADAPTERS:
            self.prototypes_ = self.estimator_.prototypes_.copy()
        self.backend_parameters_ = self._backend_parameters()
        self.converged_ = self._converged()
        self.bic_ = (
            float(self.estimator_.bic(values)) if algorithm == "gmm" else None
        )
        self.membership_ = self._membership(values)
        self.fuzzy_diagnostics_ = self._fuzzy_diagnostics()
        self.structure_ = self._structure(len(frame))
        return self

    def assign(self, frame: pd.DataFrame) -> np.ndarray:
        """Assign new rows with the fitted pipeline, without refitting.

        Raises
        ------
        CapabilityError
            If the adapter is transductive.
        """
        if not self.capability.assign:
            raise CapabilityError(
                f"{self.spec.algorithm} has no native out-of-sample "
                "assignment."
            )
        values = self._represent(frame)
        if self.spec.algorithm in ("medoids", "clara"):
            if self.prototypes_ is None:
                raise RuntimeError("Medoid prototypes are missing.")
            if self.reducer_ is not None:
                distances = pairwise_distances(values, self.prototypes_)
            else:
                distances = self.geometry_.distances(values, self.prototypes_)
            return np.asarray(distances.argmin(axis=1))
        return np.asarray(self.estimator_.predict(values))

    def loglikelihood(self, frame: pd.DataFrame) -> float:
        """Mean log-likelihood of ``frame`` in the pipeline coordinates.

        Raises
        ------
        CapabilityError
            If the adapter is not a mixture.
        """
        if self.spec.algorithm not in _MIXTURES:
            raise CapabilityError(
                "Log likelihood is only defined for mixture adapters."
            )
        return float(self.estimator_.score(self._represent(frame)))

    def _fit_labels(self, values: GeometryValues, metric: str) -> np.ndarray:
        """Fit the estimator and return an owned copy of its labels."""
        if self.spec.algorithm == "clara":
            labels = self.estimator_.fit_predict(values, self.geometry_)
        elif self.spec.algorithm in _PROTOTYPE_ADAPTERS:
            gower = self.geometry_.metric_
            if not isinstance(gower, GowerMetric) or not isinstance(
                values, pd.DataFrame
            ):
                raise CapabilityError(
                    f"{self.spec.algorithm} requires Gower geometry."
                )
            labels = self.estimator_.fit_predict(values, gower)
        else:
            labels = self.estimator_.fit_predict(
                self._model_input(values, metric)
            )
        return np.asarray(labels).copy()

    def _represent(self, frame: pd.DataFrame) -> GeometryValues:
        """Apply the fitted geometry and PCA to new rows."""
        values = self.geometry_.transform(frame)
        if self.reducer_ is not None:
            values = self.reducer_.transform(values)
        return values

    def _model_input(self, values: GeometryValues, metric: str) -> Any:
        """Return what the estimator consumes: features or distances."""
        if self.spec.algorithm == "medoids":
            if self.reducer_ is not None:
                return pairwise_distances(values)
            return self.geometry_.distances(values)
        if self.spec.is_density and metric == "precomputed":
            return self.geometry_.distances(values)
        return values

    def _backend_parameters(self) -> dict[str, Any]:
        if hasattr(self.estimator_, "get_params"):
            return dict(self.estimator_.get_params(deep=False))
        return {"max_iter": self.estimator_.max_iter, "seed": self.seed}

    def _converged(self) -> bool:
        converged = bool(getattr(self.estimator_, "converged_", True))
        if self.spec.algorithm in _ITERATION_CAPPED:
            # Reaching the iteration cap is conservatively not converged.
            if self.estimator_.n_iter_ >= self.estimator_.max_iter:
                return False
        return converged

    def _membership(self, values: GeometryValues) -> Membership | None:
        algorithm = self.spec.algorithm
        if algorithm in _MIXTURES:
            return Membership(
                "mixture_posterior",
                self.estimator_.predict_proba(values),
                tuple(range(self.spec.k or 0)),
            )
        if algorithm == "fcm":
            return Membership(
                "fuzzy_membership",
                self.estimator_.memberships_.copy(),
                tuple(range(self.spec.k or 0)),
            )
        if algorithm == "hdbscan":
            return Membership(
                "density_association_strength",
                self.estimator_.probabilities_[:, None].copy(),
                ("assigned_cluster",),
                "not_a_distribution_over_clusters",
            )
        return None

    def _fuzzy_diagnostics(self) -> dict[str, Any] | None:
        if self.spec.algorithm != "fcm":
            return None
        estimator = self.estimator_
        return {
            "xie_beni": estimator.xie_beni_,
            "fpc": estimator.fpc_,
            "partition_entropy": estimator.entropy_,
            "objective": estimator.objective_,
            "fuzzifier": self.spec.fuzzifier,
            "iterations": estimator.n_iter_,
        }

    def _structure(self, n_rows: int) -> dict[str, Any] | None:
        estimator = self.estimator_
        if self.spec.algorithm == "agglomerative":
            return {
                "kind": "ward_linkage",
                "children": estimator.children_.copy(),
                "distances": estimator.distances_.copy(),
                "leaves": n_rows,
            }
        if self.spec.algorithm == "optics":
            return {
                "kind": "optics_reachability",
                "ordering": estimator.ordering_.copy(),
                "reachability": estimator.reachability_.copy(),
                "predecessor": estimator.predecessor_.copy(),
                "core_distances": estimator.core_distances_.copy(),
            }
        return None
