"""Search planning: candidate grid, capability exclusions and budget.

Planning validates and materialises a deterministic list of candidates.
It never fits transformations or models.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ._seeding import candidate_id
from ._validation import is_integer
from .adapters import REGISTRY
from .config import (
    OBJECTIVES,
    CandidateSpec,
    GeometrySpec,
    Schema,
    SearchBudget,
    SelectionPolicy,
)
from .exceptions import CapabilityError
from .optional import optional_version
from .schema import validate_table

__all__ = [
    "DEFAULT_COMPACT_FAMILIES",
    "DEFAULT_DENSITY_GRID",
    "DEFAULT_GOWER_FAMILIES",
    "PROBABILISTIC_METRICS",
    "CandidatePlanner",
    "SearchPlan",
    "confirmation_reserve",
]

PROBABILISTIC_METRICS: tuple[str, ...] = ("bic", "heldout_loglikelihood")

DEFAULT_COMPACT_FAMILIES: tuple[str, ...] = (
    "kmeans", "agglomerative", "medoids",
)
"""Default compact families on euclidean geometry (cost-controlled)."""

DEFAULT_GOWER_FAMILIES: tuple[str, ...] = ("medoids", "kprototypes")
"""Default compact families on Gower geometry."""

DEFAULT_DENSITY_GRID: tuple[CandidateSpec, ...] = (
    *(
        CandidateSpec("dbscan", eps=eps, min_samples=m)
        for eps in (0.2, 0.5, 1.0)
        for m in (5, 10)
    ),
    *(
        CandidateSpec("optics", min_samples=5, min_cluster_size=c)
        for c in (10, 20)
    ),
    *(
        CandidateSpec("hdbscan", min_samples=m, min_cluster_size=c)
        for c in (10, 20)
        for m in (5, 10)
    ),
)
"""Default exploratory density grid."""


@dataclass(frozen=True)
class SearchPlan:
    """Validated input and the deterministic list of candidates."""

    frame: pd.DataFrame
    schema: Schema
    diagnostics: dict[str, Any]
    geometry: GeometrySpec
    policy: SelectionPolicy
    budget: SearchBudget
    candidates: tuple[CandidateSpec, ...]
    exclusions: tuple[dict[str, str], ...]

    @property
    def truncated(self) -> bool:
        """Whether ``max_candidates`` dropped any compatible candidate."""
        return any(
            e["reason"] == "max_candidates budget" for e in self.exclusions
        )


def confirmation_reserve(
    specs: Sequence[CandidateSpec], policy: SelectionPolicy
) -> int:
    """Upper bound of confirmation fits for the largest possible shortlist."""
    costs = [
        policy.subsamples
        + (policy.seed_repeats if REGISTRY[s.algorithm].stochastic else 0)
        for s in specs
    ]
    return sum(sorted(costs, reverse=True)[: policy.shortlist_size])


class CandidatePlanner:
    """Resolve geometry and build the candidate list of one objective.

    Parameters
    ----------
    objective : str
        Objective whose primary metric ranks the candidates.
    schema, geometry, selection, budget
        Optional explicit configurations; defaults are resolved here.
    candidates : iterable of CandidateSpec or None
        Explicit candidate specifications (expert mode).
    families : sequence of str or None
        Families for an automatic grid (mutually exclusive with
        ``candidates``).
    k_range : iterable of int
        Cluster counts of automatic grids.
    probabilistic_metric : str
        ``"bic"`` or ``"heldout_loglikelihood"``.
    """

    def __init__(
        self,
        objective: str,
        schema: Schema | None,
        geometry: GeometrySpec | None,
        selection: SelectionPolicy | None,
        budget: SearchBudget | None,
        candidates: Iterable[CandidateSpec] | None,
        families: Sequence[str] | None,
        k_range: Iterable[int],
        probabilistic_metric: str,
    ) -> None:
        self.objective = objective
        self.schema = schema
        self.geometry = geometry
        self.selection = selection
        self.budget = budget
        self.candidates = candidates
        self.families = families
        self.k_range = k_range
        self.probabilistic_metric = probabilistic_metric

    def plan(self, X: Any) -> SearchPlan:
        """Validate ``X`` and the configuration, then build the plan.

        Raises
        ------
        CapabilityError
            If the configuration has no compatible candidate.
        ValueError
            If the input or a parameter is invalid.
        """
        self._validate_parameters()
        table = validate_table(
            X,
            self.schema,
            allow_missing=(
                self.geometry is not None
                and self.geometry.missing == "pairwise"
            ),
        )
        geometry = self._resolve_geometry(table.schema)
        policy = self.selection or SelectionPolicy()
        budget = self.budget or SearchBudget()
        specs = self._deduplicated(self._specifications(geometry))
        included, exclusions = self._apply_capabilities(
            specs, geometry, len(table.frame)
        )
        self._check_comparability(included)
        planned = included[: budget.max_candidates]
        exclusions += [
            {
                "candidate_id": candidate_id(s),
                "reason": "max_candidates budget",
            }
            for s in included[budget.max_candidates:]
        ]
        return SearchPlan(
            frame=table.frame,
            schema=table.schema,
            diagnostics=table.diagnostics,
            geometry=geometry,
            policy=policy,
            budget=budget,
            candidates=tuple(planned),
            exclusions=tuple(exclusions),
        )

    def _validate_parameters(self) -> None:
        if self.objective not in OBJECTIVES:
            raise CapabilityError(
                "Supported objectives: compact, probabilistic, density or "
                "fuzzy."
            )
        if self.probabilistic_metric not in PROBABILISTIC_METRICS:
            raise ValueError(
                "probabilistic_metric must be bic or heldout_loglikelihood."
            )
        if self.candidates is not None and self.families is not None:
            raise ValueError(
                "Choose families for automatic grids or candidates for "
                "explicit specifications, not both."
            )
        if self.families is not None:
            self._validate_families(self.families)
        if self.candidates is None:
            self._validate_k_range()

    def _validate_families(self, families: Sequence[str]) -> None:
        if (
            isinstance(families, str)
            or not families
            or any(f not in REGISTRY for f in families)
        ):
            raise ValueError(
                "families must be a nonempty sequence of known adapter names."
            )
        if any(REGISTRY[f].objective != self.objective for f in families):
            raise CapabilityError(
                "Automatic families must belong to the requested objective."
            )

    def _validate_k_range(self) -> None:
        if isinstance(self.k_range, (str, bytes)):
            raise ValueError("k_range must be a sequence of integers >= 2.")
        values = list(self.k_range)
        if self.objective != "density" and (
            not values
            or any(not is_integer(k) or int(k) < 2 for k in values)
        ):
            raise ValueError(
                "k_range must be a nonempty sequence of integers >= 2."
            )

    def _resolve_geometry(self, schema: Schema) -> GeometrySpec:
        if self.geometry is not None:
            geometry = self.geometry
        elif schema.nonnumeric:
            geometry = GeometrySpec("gower", "none")
        else:
            geometry = GeometrySpec()
        if geometry.kind == "euclidean" and schema.nonnumeric:
            raise CapabilityError(
                "Use Gower geometry for nominal, ordinal, binary or mixed "
                "inputs."
            )
        unknown = [c for c, _ in geometry.weights if c not in schema.features]
        if unknown:
            raise ValueError(
                f"Weights name columns that are not features: {unknown}."
            )
        if (
            self.objective in ("probabilistic", "fuzzy")
            and geometry.kind != "euclidean"
        ):
            raise CapabilityError(
                "Mixtures and FCM require euclidean numeric features."
            )
        return geometry

    def _default_families(self, geometry: GeometrySpec) -> tuple[str, ...]:
        if self.families is not None:
            return tuple(self.families)
        if self.objective == "compact":
            if geometry.kind == "gower":
                return DEFAULT_GOWER_FAMILIES
            return DEFAULT_COMPACT_FAMILIES
        if self.objective == "probabilistic":
            if self.probabilistic_metric == "bic":
                return ("gmm",)
            return ("gmm", "bgmm")
        return ("fcm",)

    def _specifications(self, geometry: GeometrySpec) -> list[CandidateSpec]:
        if self.candidates is not None:
            specs = list(self.candidates)
        elif self.objective == "density":
            specs = [
                s for s in DEFAULT_DENSITY_GRID
                if self.families is None or s.algorithm in self.families
            ]
        else:
            families = self._default_families(geometry)
            # Interleave families for each k so truncation stays balanced.
            specs = [
                CandidateSpec(algorithm, int(k))
                for k in self.k_range
                for algorithm in families
            ]
        if not specs or not all(isinstance(s, CandidateSpec) for s in specs):
            raise ValueError("Provide at least one CandidateSpec.")
        return specs

    @staticmethod
    def _deduplicated(specs: list[CandidateSpec]) -> list[CandidateSpec]:
        """Drop repeated specifications, keeping first-seen order."""
        return list({candidate_id(s): s for s in specs}.values())

    def _exclusion_reason(
        self, spec: CandidateSpec, geometry: GeometrySpec, n_rows: int
    ) -> str | None:
        capability = REGISTRY[spec.algorithm]
        if capability.objective != self.objective:
            return "incompatible objective"
        if geometry.kind not in capability.geometries:
            return "incompatible geometry"
        if spec.pca_variance is not None and geometry.kind != "euclidean":
            return "PCA requires euclidean numeric features"
        if spec.algorithm == "bgmm" and self.probabilistic_metric == "bic":
            return (
                "BayesianGaussianMixture has no comparable BIC; choose "
                "heldout_loglikelihood"
            )
        feature = capability.optional_feature
        if feature is not None and optional_version(feature) is None:
            return (
                "optional dependency missing or incompatible; install "
                f"clustelia[{feature}]"
            )
        if spec.algorithm in ("hdbscan", "optics") and max(
            spec.min_samples, spec.min_cluster_size
        ) > n_rows:
            return "density minimum sample/cluster size exceeds available rows"
        if spec.k is not None and spec.k >= n_rows:
            return "k must be less than number of rows"
        return None

    def _apply_capabilities(
        self,
        specs: list[CandidateSpec],
        geometry: GeometrySpec,
        n_rows: int,
    ) -> tuple[list[CandidateSpec], list[dict[str, str]]]:
        included: list[CandidateSpec] = []
        exclusions: list[dict[str, str]] = []
        for spec in specs:
            reason = self._exclusion_reason(spec, geometry, n_rows)
            if reason is None:
                included.append(spec)
            else:
                exclusions.append(
                    {"candidate_id": candidate_id(spec), "reason": reason}
                )
        if not included:
            raise CapabilityError(f"All candidates excluded: {exclusions}")
        return included, exclusions

    def _check_comparability(self, included: list[CandidateSpec]) -> None:
        """Reject runs whose primary metric cannot rank the candidates."""
        if self.objective in ("probabilistic", "fuzzy") and len(
            {s.pca_variance for s in included}
        ) > 1:
            raise CapabilityError(
                "This primary metric cannot rank different representations; "
                "run separate experiments."
            )
        if self.objective == "fuzzy" and len(
            {s.fuzzifier for s in included}
        ) > 1:
            raise CapabilityError(
                "Fix fuzzifier within an objective; compare its sensitivity "
                "in separate experiments."
            )
