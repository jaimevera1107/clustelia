"""Immutable search specifications.

These objects describe *what* to search; they never hold fitted state, so
they can be shared, hashed and cloned safely. Values are validated and
normalised at construction time.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

from ._validation import (
    require_finite,
    require_int,
    require_nonnegative,
    require_open_unit,
    require_positive,
    require_unit_interval,
)

__all__ = [
    "ALGORITHMS",
    "DENSITY_ALGORITHMS",
    "OBJECTIVES",
    "CandidateSpec",
    "candidate_parameters",
    "CostTiebreak",
    "KSelection",
    "NullScope",
    "GeometryKind",
    "MissingPolicy",
    "GeometrySpec",
    "Objective",
    "Scaling",
    "Schema",
    "SearchBudget",
    "SelectionPolicy",
]

Objective = Literal["compact", "probabilistic", "density", "fuzzy"]
GeometryKind = Literal["euclidean", "gower"]
Scaling = Literal["standard", "robust", "none"]
MissingPolicy = Literal["error", "pairwise"]
CovarianceType = Literal["full", "tied", "diag", "spherical"]
CostTiebreak = Literal["fit_count", "fit_seconds"]
NullScope = Literal["grid", "family"]
KSelection = Literal["auto", "gap", "silhouette"]

OBJECTIVES: tuple[str, ...] = get_args(Objective)
ALGORITHMS: tuple[str, ...] = (
    "kmeans",
    "medoids",
    "clara",
    "agglomerative",
    "gmm",
    "dbscan",
    "minibatch",
    "birch",
    "bisecting",
    "spectral",
    "optics",
    "hdbscan",
    "fcm",
    "bgmm",
    "kprototypes",
)
DENSITY_ALGORITHMS: frozenset[str] = frozenset({"dbscan", "optics", "hdbscan"})


def _as_column_tuple(name: str, columns: Iterable[Hashable]) -> tuple:
    """Convert a column collection to a tuple, rejecting bare strings.

    A bare string is iterable, so ``Schema(numeric="age")`` would silently
    become ``("a", "g", "e")`` without this check.
    """
    if isinstance(columns, (str, bytes)):
        raise ValueError(
            f"Schema.{name} must be a sequence of column names, not a "
            "single string."
        )
    return tuple(columns)


def _as_ordinal_tuple(ordinal: Any) -> tuple:
    """Normalise ordinal declarations to ``((column, levels), ...)``.

    Accepts a mapping ``{column: ordered levels}`` or an iterable of
    ``(column, levels)`` pairs. Levels are stored as tuples so the schema
    stays hashable.
    """
    if isinstance(ordinal, Mapping):
        items = list(ordinal.items())
    elif isinstance(ordinal, (str, bytes)):
        raise ValueError(
            "Schema.ordinal must map columns to ordered levels."
        )
    else:
        items = [tuple(item) for item in ordinal]
    normalised: list[tuple[Hashable, tuple]] = []
    for item in items:
        if len(item) != 2:
            raise ValueError(
                "Schema.ordinal entries must be (column, levels) pairs."
            )
        column, levels = item
        if isinstance(levels, (str, bytes)):
            raise ValueError(
                f"Ordinal levels of {column!r} must be a sequence, not a "
                "single string."
            )
        levels = tuple(levels)
        if len(levels) < 2 or len(set(levels)) != len(levels):
            raise ValueError(
                f"Ordinal column {column!r} needs at least two distinct, "
                "ordered levels."
            )
        normalised.append((column, levels))
    return tuple(normalised)


@dataclass(frozen=True)
class Schema:
    """Explicit role of every input column.

    Parameters
    ----------
    numeric : sequence of hashable
        Real-valued feature columns.
    categorical : sequence of hashable
        Nominal feature columns (no order is assumed). Symmetric binary
        variables belong here.
    identifiers : sequence of hashable
        Entity identifier columns; validated for uniqueness and excluded
        from fitting.
    ordinal : mapping of hashable to sequence
        Ordered feature columns and their levels, lowest first. Values
        outside the declared levels raise an error.
    binary : sequence of hashable
        Asymmetric binary feature columns coded 0/1 (or False/True). Under
        Gower, a joint absence (0, 0) carries no information and is left
        out of the comparison, following Gower (1971).
    """

    numeric: tuple = ()
    categorical: tuple = ()
    identifiers: tuple = ()
    ordinal: tuple = ()
    binary: tuple = ()

    def __post_init__(self) -> None:
        for name in ("numeric", "categorical", "identifiers", "binary"):
            columns = _as_column_tuple(name, getattr(self, name))
            object.__setattr__(self, name, columns)
        object.__setattr__(self, "ordinal", _as_ordinal_tuple(self.ordinal))
        declared = self.features + self.identifiers
        if len(set(declared)) != len(declared):
            raise ValueError("Schema columns must be unique and disjoint.")

    @property
    def ordinal_columns(self) -> tuple:
        """Ordinal feature columns in declaration order."""
        return tuple(column for column, _ in self.ordinal)

    @property
    def ordinal_levels(self) -> dict[Hashable, tuple]:
        """Declared ordered levels of every ordinal column."""
        return dict(self.ordinal)

    @property
    def nonnumeric(self) -> tuple:
        """Feature columns that are not real-valued."""
        return self.categorical + self.ordinal_columns + self.binary

    @property
    def features(self) -> tuple:
        """Feature columns: numeric, categorical, ordinal, binary."""
        return self.numeric + self.nonnumeric

    @property
    def columns(self) -> tuple:
        """Every declared column, identifiers included."""
        return self.features + self.identifiers


def _as_weight_tuple(weights: Any) -> tuple:
    """Normalise variable weights to ``((column, weight), ...)``."""
    if isinstance(weights, Mapping):
        items = list(weights.items())
    elif isinstance(weights, (str, bytes)):
        raise ValueError("weights must map columns to positive numbers.")
    else:
        items = [tuple(item) for item in weights]
    normalised = tuple(
        (column, require_positive(f"weight of {column!r}", value))
        for column, value in items
    )
    if len({column for column, _ in normalised}) != len(normalised):
        raise ValueError("Each weighted column must appear once.")
    return normalised


@dataclass(frozen=True)
class GeometrySpec:
    """Reference geometry used to compare all candidates.

    Parameters
    ----------
    kind : {"euclidean", "gower"}
        Distance used for every reference metric.
    scaling : {"standard", "robust", "none"}
        Numeric scaling for euclidean geometry. Gower geometry uses fitted
        numeric ranges and therefore requires ``"none"``.
    weights : mapping of hashable to float
        Expert weight of each original variable under Gower (default 1 for
        every undeclared feature). Weights must be positive and finite;
        they are relative, so only their ratios matter.
    gower_backend : {"stable", "vectorized"}
        ``"stable"`` preserves the reference floating-point summation order.
        ``"vectorized"`` uses bounded feature batches and may differ by
        floating-point rounding, potentially changing exact ties.
    incomparable : {"error", "max", "zero"}
        Treatment for pairs without common comparable weight, or below
        ``min_comparable_share``. Default raises, ``max`` assigns distance
        1 (penalty), and ``zero`` restores the legacy/unsafe semantics.
    min_comparable_share : float
        Minimum shared applicable weight / all feature weights, in [0,1].
        A threshold of 0 only rejects pairs with exactly zero coverage.
    null_blocks : tuple of tuples
        Optional blocks of Gower features permuted together under the
        null-reference test. Preserve known dependencies within each block
        (e.g. one-hot dummy groups). Default empty: permute independently.
    missing : {"error", "pairwise"}
        Missing-value policy. ``"error"`` (default) rejects missing
        features. ``"pairwise"`` (Gower only) compares each pair of rows on
        the variables observed in both, as in Gower (1971); every row must
        have at least one observed feature.
    """

    kind: GeometryKind = "euclidean"
    scaling: Scaling = "standard"
    weights: tuple = ()
    missing: MissingPolicy = "error"
    gower_backend: str = "stable"
    incomparable: str = "error"
    min_comparable_share: float = 0.0
    null_blocks: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", _as_weight_tuple(self.weights))
        blocks = tuple(tuple(block) for block in self.null_blocks)
        if any(not block for block in blocks):
            raise ValueError("null_blocks must contain nonempty groups")
        members = [item for block in blocks for item in block]
        if len(members) != len(set(members)):
            raise ValueError("null_blocks cannot overlap")
        object.__setattr__(self, "null_blocks", blocks)
        if blocks and self.kind != "gower":
            raise ValueError("null_blocks apply only to Gower references")
        if self.incomparable not in ("error", "max", "zero"):
            raise ValueError("incomparable must be error, max or zero.")
        if (not isinstance(self.min_comparable_share, (int, float)) or
                not 0 <= self.min_comparable_share <= 1):
            raise ValueError("min_comparable_share must be in [0, 1].")
        if self.kind != "gower" and (
            self.incomparable != "error" or self.min_comparable_share != 0
        ):
            raise ValueError("Comparability settings require Gower geometry.")
        if self.gower_backend not in ("stable", "vectorized"):
            raise ValueError("gower_backend must be stable or vectorized.")
        if self.gower_backend != "stable" and self.kind != "gower":
            raise ValueError("vectorized Gower requires kind='gower'.")
        if self.weights and self.kind != "gower":
            raise ValueError(
                "Variable weights are defined for Gower geometry only."
            )
        if self.kind not in get_args(GeometryKind):
            raise ValueError("Supported geometry: euclidean or gower.")
        if self.scaling not in get_args(Scaling):
            raise ValueError("Supported scaling: standard, robust or none.")
        if self.kind == "gower" and self.scaling != "none":
            raise ValueError(
                "Gower uses fitted numeric ranges; set scaling='none'."
            )
        if self.missing not in get_args(MissingPolicy):
            raise ValueError("Supported missing policy: error or pairwise.")
        if self.missing == "pairwise" and self.kind != "gower":
            raise ValueError(
                "Pairwise deletion of missing values is defined for Gower "
                "geometry only."
            )


@dataclass(frozen=True)
class SelectionPolicy:
    """Operating thresholds of the selection policy.

    The stability thresholds are engineering rules checked on the
    synthetic benchmark in ``tools/benchmark.py``; they are not statistical
    guarantees. The null-reference screen (``null_replicates``) is
    calibrated only under an exchangeable reference design, which
    data-dependent fitted reference families do not guarantee in general.

    Parameters
    ----------
    version : str
        Methodology label recorded in every result.
    min_coverage : float
        Minimum fraction of observations assigned to a cluster.
    min_stability : float
        Minimum median ARI on every applicable stability axis.
    min_evaluable_fraction : float
        Minimum fraction of replicas yielding an evaluable ARI.
    silhouette_tolerance : float
        Practical-equivalence band for the compact objective.
    shortlist_size : int
        Maximum number of candidates sent to confirmation.
    subsamples : int
        Subsampling replicas per shortlisted candidate.
    seed_repeats : int
        Extra-seed replicas per shortlisted stochastic candidate.
    sample_fraction : float
        Fraction of rows drawn without replacement per subsample.
    min_common : int
        Floor of the common assigned support for a stability comparison.
    duplicate_ari : float
        ARI above which two partitions are considered redundant.
    density_tolerance, fuzzy_tolerance : float or None
        Practical-equivalence bands for the density and fuzzy objectives.
        ``None`` keeps several admissible alternatives as ``ambiguous``.
    min_assignment_agreement : float
        Minimum agreement of the assigned/noise status per replica.
    null_replicates : int
        Null-reference replicates per family for the compact objective. A
        recommendation must have a reference silhouette above every
        replicate. ``1 / (B + 1)`` is conditional Monte Carlo rank
        resolution under exchangeability, not a general error bound.
        ``0`` disables the test and
        restores the P0 behaviour.
    null_scope : {"grid", "family"}
        Candidates refitted on each null dataset. ``"grid"`` (default)
        refits every explored candidate and mirrors the whole selection;
        ``"family"`` refits only the preferred candidate's family, which is
        cheaper but ignores the selection across families and is optimistic
        on structureless data.
    k_selection : {"auto", "gap", "silhouette"}
        How the compact objective chooses the number of clusters. ``"gap"``
        picks ``k`` with the gap statistic and its one-standard-error rule,
        reusing the null-reference fits, then confirms only candidates
        with that ``k`` and prefers the best silhouette among them.
        ``"silhouette"`` is the P0/P1 rule (silhouette band and stability
        tie-breaks over every ``k``). ``"auto"`` uses ``"gap"`` when the
        null test runs on the whole grid with at least two replicates, and
        ``"silhouette"`` otherwise.
    cost_tiebreak : {"fit_count", "fit_seconds"}
        Cost used to break ties after stability and coverage. The default,
        ``"fit_count"``, is deterministic and expressed in the same unit as
        :attr:`SearchBudget.max_fits`. ``"fit_seconds"`` reproduces the
        wall-clock rule of 0.1.0a2, whose winner may vary between runs.
    """

    version: str = "P2-alpha1"
    min_coverage: float = 0.8
    min_stability: float = 0.7
    min_evaluable_fraction: float = 0.8
    silhouette_tolerance: float = 0.02
    shortlist_size: int = 5
    subsamples: int = 10
    seed_repeats: int = 3
    sample_fraction: float = 0.8
    min_common: int = 20
    duplicate_ari: float = 0.98
    density_tolerance: float | None = None
    fuzzy_tolerance: float | None = None
    min_assignment_agreement: float = 0.8
    null_replicates: int = 9
    null_scope: NullScope = "grid"
    k_selection: KSelection = "auto"
    cost_tiebreak: CostTiebreak = "fit_count"

    @classmethod
    def p0(cls, **kwargs: Any) -> SelectionPolicy:
        """Return the P0 policy (no null-reference test), with overrides."""
        kwargs.setdefault("k_selection", "silhouette")
        kwargs.setdefault("version", "P0-alpha3")
        kwargs.setdefault("null_replicates", 0)
        return cls(**kwargs)

    def __post_init__(self) -> None:
        for key in (
            "min_coverage",
            "min_stability",
            "min_evaluable_fraction",
            "duplicate_ari",
            "min_assignment_agreement",
        ):
            self._set(key, require_unit_interval(key, getattr(self, key)))
        self._set(
            "sample_fraction",
            require_open_unit("sample_fraction", self.sample_fraction),
        )
        for key in ("shortlist_size", "subsamples", "seed_repeats",
                    "min_common"):
            self._set(key, require_int(key, getattr(self, key), 1))
        self._set(
            "null_replicates",
            require_int("null_replicates", self.null_replicates, 0),
        )
        self._set(
            "silhouette_tolerance",
            require_nonnegative(
                "silhouette_tolerance", self.silhouette_tolerance
            ),
        )
        for key in ("density_tolerance", "fuzzy_tolerance"):
            value = getattr(self, key)
            if value is not None:
                self._set(key, require_nonnegative(key, value))
        if self.null_scope not in get_args(NullScope):
            raise ValueError("null_scope must be grid or family.")
        if self.k_selection not in get_args(KSelection):
            raise ValueError("k_selection must be auto, gap or silhouette.")
        if self.k_selection == "gap" and not self._gap_available():
            raise ValueError(
                "k_selection='gap' needs null_scope='grid' and "
                "null_replicates >= 2."
            )
        if self.cost_tiebreak not in get_args(CostTiebreak):
            raise ValueError("cost_tiebreak must be fit_count or fit_seconds.")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be a nonempty string.")

    @property
    def uses_gap(self) -> bool:
        """Whether the number of clusters is chosen by the gap statistic."""
        if self.k_selection == "silhouette":
            return False
        return self._gap_available()

    def _gap_available(self) -> bool:
        return self.null_scope == "grid" and self.null_replicates >= 2

    def _set(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SearchBudget:
    """Resource limits shared by exploration and confirmation.

    Parameters
    ----------
    max_fits : int
        Total pipeline fits, failed attempts and replicas included.
    max_candidates : int
        Unique candidate specifications explored.
    timeout_s : float
        Soft deadline in seconds, checked between fits.
    max_pairwise_mb : float
        Preventive estimate limit for dense pairwise matrices (MiB).
    metric_sample_size : int
        Rows of the common sample used by reference metrics.
    workers : int or None
        Worker processes for the null-reference stage, which refits every
        explored candidate on structureless data and is most of the cost.
        ``None`` (default) chooses automatically: 1 below 1000 rows, else
        ``min(8, CPU count)``. ``1`` runs sequentially, ``k > 1`` uses at
        most ``k`` processes and ``-1`` all CPUs. Results, candidate
        identifiers, fit counts and seeds do not depend on this value;
        only wall-clock time and memory (each worker holds a copy of the
        data) do. Fits are counted in the parent and the budgets are
        checked before each batch.
    """

    max_fits: int = 320
    max_candidates: int = 40
    timeout_s: float = 600
    max_pairwise_mb: float = 256
    metric_sample_size: int = 1000
    workers: int | None = None

    def __post_init__(self) -> None:
        for key in ("max_fits", "max_candidates"):
            value = require_int(key, getattr(self, key), 1)
            object.__setattr__(self, key, value)
        object.__setattr__(
            self,
            "metric_sample_size",
            require_int("metric_sample_size", self.metric_sample_size, 3),
        )
        for key in ("timeout_s", "max_pairwise_mb"):
            object.__setattr__(
                self, key, require_positive(key, getattr(self, key))
            )
        if self.workers is not None and self.workers != -1:
            object.__setattr__(
                self, "workers", require_int("workers", self.workers, 1)
            )


@dataclass(frozen=True)
class CandidateSpec:
    """One algorithm configuration to evaluate.

    Only the parameters relevant to ``algorithm`` are used by its adapter;
    all parameters are validated so that invalid values fail early.

    Parameters
    ----------
    algorithm : str
        Adapter name; see :data:`ALGORITHMS`.
    k : int or None
        Number of clusters. Must be ``None`` for density algorithms.
    pca_variance : float or None
        Explained-variance target of an optional PCA fitting space.
    eps, min_samples : float, int
        DBSCAN neighbourhood radius and core size (``min_samples`` is also
        used by OPTICS and HDBSCAN).
    covariance_type : {"full", "tied", "diag", "spherical"}
        Mixture covariance structure.
    min_cluster_size, xi : int, float
        OPTICS/HDBSCAN cluster size and OPTICS steepness.
    threshold : float
        BIRCH subcluster radius.
    gamma : float
        Spectral RBF kernel coefficient.
    batch_size : int
        MiniBatchKMeans batch size.
    fuzzifier : float
        Fuzzy C-Means exponent ``m`` (> 1).
    concentration_prior : float
        Dirichlet-process weight concentration of the Bayesian mixture.
    clara_sample_size, clara_samples : int
        Subsample target and full-data trial count of the opt-in CLARA
        medoids search; its full-data objective uses chunked distances.
    """

    algorithm: str
    k: int | None = None
    pca_variance: float | None = None
    eps: float = 0.5
    min_samples: int = 5
    covariance_type: CovarianceType = "full"
    min_cluster_size: int = 10
    xi: float = 0.05
    threshold: float = 0.5
    gamma: float = 1.0
    batch_size: int = 256
    fuzzifier: float = 2.0
    concentration_prior: float = 1.0
    clara_sample_size: int = 64
    clara_samples: int = 5

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        self._validate_k()
        if self.pca_variance is not None:
            self._set(
                "pca_variance",
                require_open_unit("pca_variance", self.pca_variance),
            )
        try:
            self._set("eps", require_positive("eps", self.eps))
            self._set(
                "min_samples", require_int("min_samples", self.min_samples, 1)
            )
        except ValueError as exc:
            raise ValueError(f"Invalid DBSCAN parameters: {exc}") from None
        if self.covariance_type not in get_args(CovarianceType):
            raise ValueError("Invalid mixture covariance_type.")
        self._set(
            "min_cluster_size",
            require_int("min_cluster_size", self.min_cluster_size, 2),
        )
        self._set("batch_size", require_int("batch_size", self.batch_size, 1))
        self._set("clara_sample_size", require_int(
            "clara_sample_size", self.clara_sample_size, 2
        ))
        self._set("clara_samples", require_int(
            "clara_samples", self.clara_samples, 1
        ))
        self._set("xi", require_open_unit("xi", self.xi))
        fuzzifier = require_finite("fuzzifier", self.fuzzifier)
        if fuzzifier <= 1:
            raise ValueError("fuzzifier must be finite and greater than 1.")
        self._set("fuzzifier", fuzzifier)
        for key in ("threshold", "gamma", "concentration_prior"):
            self._set(key, require_positive(key, getattr(self, key)))

    @property
    def is_density(self) -> bool:
        """Whether the adapter discovers the number of clusters itself."""
        return self.algorithm in DENSITY_ALGORITHMS

    def _validate_k(self) -> None:
        if self.is_density:
            if self.k is not None:
                raise ValueError(
                    "Density adapters discover their clusters; k must be "
                    "None."
                )
            return
        self._set("k", require_int("k", self.k, 2))

    def _set(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)


def candidate_parameters(spec: CandidateSpec) -> dict[str, Any]:
    """Canonical parameter record, preserving existing model identities.

    CLARA-specific fields never change candidate hashes or serialized
    parameters of other algorithms. New CLARA candidates include both.
    """
    from dataclasses import asdict

    params = asdict(spec)
    if spec.algorithm != "clara":
        params.pop("clara_sample_size")
        params.pop("clara_samples")
    return params
