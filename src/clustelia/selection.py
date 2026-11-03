"""Selection policy: admissibility, shortlist and preference.

The thresholds are operating rules, not statistical guarantees.
No universal score ranks different objectives or geometries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .config import SelectionPolicy
from .records import CandidateRow

if TYPE_CHECKING:
    from .results import Solution

__all__ = [
    "build_shortlist",
    "choose",
    "deduplicate",
    "exploration_rejections",
    "prefer",
    "quality",
]

_STABILITY_AXES: tuple[str, ...] = ("composition", "randomness")


def quality(row: CandidateRow, objective: str) -> float:
    """Return the primary score, oriented so that larger is better."""
    if "primary_score" in row:
        return float(row["primary_score"])
    if objective == "probabilistic":
        return -float(row["bic"])
    return float(row["silhouette"])


def exploration_rejections(
    row: CandidateRow,
    converged: bool,
    policy: SelectionPolicy,
    objective: str,
    n_rows: int,
) -> list[str]:
    """Return the reasons a fitted candidate cannot be shortlisted."""
    reasons: list[str] = []
    if not converged:
        reasons.append("backend did not converge")
    if row["coverage"] < policy.min_coverage:
        reasons.append("below coverage threshold")
    if not 2 <= row["n_clusters"] < n_rows:
        reasons.append("degenerate partition")
    if row["silhouette"] is None and objective == "compact":
        reasons.append("no evaluable reference silhouette")
    score = row["primary_score"]
    if score is None or not np.isfinite(score):
        reasons.append("no evaluable primary score")
    return reasons


def deduplicate(
    rows: Sequence[CandidateRow],
    solutions: Mapping[str, Solution],
    threshold: float,
) -> list[CandidateRow]:
    """Drop near-identical partitions, keeping the first (best) of each.

    Two partitions are redundant when their noise sets coincide and their
    ARI on assigned rows is at least ``threshold``.
    """
    selected: list[CandidateRow] = []
    for row in rows:
        labels = solutions[row["candidate_id"]].labels_
        if not any(
            _redundant(labels, solutions[other["candidate_id"]].labels_,
                       threshold)
            for other in selected
        ):
            selected.append(row)
    return selected


def _redundant(left: np.ndarray, right: np.ndarray, threshold: float) -> bool:
    if not np.array_equal(left < 0, right < 0):
        return False
    keep = left >= 0
    return bool(
        keep.any()
        and adjusted_rand_score(left[keep], right[keep]) >= threshold
    )


def build_shortlist(
    rows: Sequence[CandidateRow], size: int
) -> list[CandidateRow]:
    """Take one representative per family first, then fill by order.

    ``rows`` must already be sorted by decreasing quality and deduplicated.
    """
    shortlist: list[CandidateRow] = []
    families: set[str] = set()
    for row in rows:
        if row["algorithm"] not in families:
            shortlist.append(row)
            families.add(row["algorithm"])
    chosen = {row["candidate_id"] for row in shortlist}
    shortlist += [row for row in rows if row["candidate_id"] not in chosen]
    return shortlist[:size]


def _stability_reasons(
    row: CandidateRow, policy: SelectionPolicy
) -> list[str]:
    reasons: list[str] = []
    for axis in _STABILITY_AXES:
        stats = row[axis]
        if stats is None:  # Deterministic adapters have no seed axis.
            continue
        if stats["evaluable_fraction"] < policy.min_evaluable_fraction:
            reasons.append(f"{axis}: insufficient evaluable replicas")
        if stats["median"] is None or stats["median"] < policy.min_stability:
            reasons.append(f"{axis}: below stability threshold")
    return reasons


def _pareto_frontier(
    admissible: Sequence[CandidateRow], objective: str
) -> list[str]:
    """Non-dominated candidates on quality, stability and coverage."""

    def vector(row: CandidateRow) -> tuple[float, float, float]:
        """Objective vector of a candidate (larger is better)."""
        return (
            quality(row, objective),
            row["composition"]["median"],
            row["coverage"],
        )

    def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        """Whether ``a`` Pareto-dominates ``b``."""
        return all(x >= y for x, y in zip(a, b, strict=True)) and any(
            x > y for x, y in zip(a, b, strict=True)
        )

    vectors = {row["candidate_id"]: vector(row) for row in admissible}
    return [
        cid for cid, v in vectors.items()
        if not any(dominates(other, v) for other in vectors.values())
    ]


def _tolerance(objective: str, policy: SelectionPolicy) -> float | None:
    """Practical-equivalence band; ``None`` means no validated band."""
    tolerances: dict[str, float | None] = {
        "compact": policy.silhouette_tolerance,
        # Comparable criteria (BIC, held-out likelihood) use exact optima.
        "probabilistic": 0.0,
        "density": policy.density_tolerance,
        "fuzzy": policy.fuzzy_tolerance,
    }
    return tolerances[objective]


def _tiebreak_key(
    row: CandidateRow, policy: SelectionPolicy
) -> tuple[Any, ...]:
    """Order equivalent candidates: stability, coverage, cost, id."""
    if policy.cost_tiebreak == "fit_seconds":
        cost: float = row["fit_seconds"]
    else:
        cost = row.get("fits_used", 0)
    return (
        -row["composition"]["median"],
        -row["coverage"],
        cost,
        row["candidate_id"],
    )


def choose(
    rows: Sequence[CandidateRow], objective: str, policy: SelectionPolicy
) -> tuple[str | None, list[str], list[str]]:
    """Apply the confirmation filters and the preference rule.

    Confirmed rows failing a stability filter receive rejection reasons
    (the rows are updated in place).

    Returns
    -------
    winner : str or None
        Preferred candidate, or ``None`` when no defensible preference
        exists.
    frontier : list of str
        Pareto-frontier candidate identifiers.
    admissible : list of str
        Identifiers of every admissible candidate.
    """
    admissible: list[CandidateRow] = []
    for row in rows:
        if not row.get("confirmed"):
            continue
        reasons = _stability_reasons(row, policy)
        row.setdefault("rejection_reasons", []).extend(reasons)
        if not reasons:
            admissible.append(row)
    if not admissible:
        return None, [], []
    admissible_ids = [row["candidate_id"] for row in admissible]
    frontier = _pareto_frontier(admissible, objective)
    return prefer(admissible, objective, policy), frontier, admissible_ids


def prefer(
    admissible: Sequence[CandidateRow],
    objective: str,
    policy: SelectionPolicy,
) -> str | None:
    """Apply the preference rule to rows that are already admissible.

    Returns ``None`` when there is no row, or when the objective has no
    validated tolerance and several rows remain.
    """
    if not admissible:
        return None
    tolerance = _tolerance(objective, policy)
    if tolerance is None and len(admissible) > 1:
        return None
    band = 0.0 if tolerance is None else tolerance
    best = max(quality(row, objective) for row in admissible)
    close = [r for r in admissible if best - quality(r, objective) <= band]
    winner = min(close, key=lambda r: _tiebreak_key(r, policy))
    return str(winner["candidate_id"])
