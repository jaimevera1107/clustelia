"""Choice of the number of clusters with the gap statistic.

The silhouette rewards the top level of a hierarchy: two close blobs score
higher merged than apart, so silhouette-based selection tends to merge
nearby clusters. The gap statistic (Tibshirani, Walther and Hastie, 2001)
compares the within-cluster dispersion of each partition with the
dispersion the same pipeline reaches on structureless reference data:

``Gap(k) = mean_b log W*_kb - log W_k``

where ``W`` is the pooled within-cluster dispersion in the reference
geometry. The reference fits are those already run by the null-reference
test, so the statistic costs no extra fit. For each ``k`` the candidate
with the largest gap represents that ``k``; the selected ``k`` is the
smallest one with ``Gap(k) >= Gap(k') - s_k'``, where ``k'`` is the next
explored ``k`` and ``s = sd(log W*) * sqrt(1 + 1/B)`` (the authors'
one-standard-error rule).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .records import CandidateRow

__all__ = ["GapEntry", "gap_table", "select_k"]

_MIN_REPLICATES: int = 2


@dataclass(frozen=True)
class GapEntry:
    """Gap of the best candidate for one number of clusters.

    Attributes
    ----------
    k : int
        Number of clusters.
    candidate_id : str
        Candidate with the largest gap at this ``k``.
    gap : float
        ``mean log W*`` over replicates minus ``log W`` on the data.
    standard_error : float
        ``sd(log W*) * sqrt(1 + 1/B)``.
    """

    k: int
    candidate_id: str
    gap: float
    standard_error: float

    def as_record(self) -> dict[str, object]:
        """Plain-dictionary form stored in the result metadata."""
        return asdict(self)


def _entry(
    row: CandidateRow, null: Sequence[float | None]
) -> GapEntry | None:
    """Gap of one candidate, or ``None`` when it cannot be computed."""
    observed = row.get("within_dispersion")
    # Zero observed dispersion is a legitimate exact partition, not an
    # unavailable statistic. If all null dispersions are strictly positive,
    # its Gap is +inf, the limiting value as W approaches zero from above.
    # Non-positive/NaN null references cannot be used in a log statistic.
    if (observed is None or not np.isfinite(observed) or observed < 0
            or len(null) < _MIN_REPLICATES):
        return None
    if any(v is None or not np.isfinite(v) or v <= 0 for v in null):
        return None
    values = np.asarray(null, dtype=float)
    logs = np.log(values)
    return GapEntry(
        k=int(row["parameters"]["k"]),
        candidate_id=str(row["candidate_id"]),
        gap=(float("inf") if observed == 0 else
             float(logs.mean() - np.log(observed))),
        # Population SD (ddof=0) is the explicit historical Clustelia
        # convention; R clusGap uses ddof=1. See VALIDATION.md.
        standard_error=float(logs.std(ddof=0) * np.sqrt(1 + 1 / len(logs))),
    )


def gap_table(
    rows: Sequence[CandidateRow],
    null_dispersions: Mapping[str, Sequence[float | None]],
) -> list[GapEntry]:
    """Best gap per ``k`` over the explored candidates, sorted by ``k``.

    Parameters
    ----------
    rows : sequence of dict
        Candidate records with ``within_dispersion`` and ``parameters``.
    null_dispersions : mapping
        Null dispersions per candidate identifier, one per replicate.
    """
    best: dict[int, GapEntry] = {}
    for row in rows:
        null = null_dispersions.get(row["candidate_id"])
        if row["status"] != "fitted" or null is None:
            continue
        entry = _entry(row, null)
        if entry is not None and (
            entry.k not in best or entry.gap > best[entry.k].gap
        ):
            best[entry.k] = entry
    return [best[k] for k in sorted(best)]


def select_k(table: Sequence[GapEntry]) -> int | None:
    """Smallest ``k`` satisfying the one-standard-error rule.

    Returns ``None`` for an empty table and the only ``k`` of a
    single-entry table.
    """
    for position, entry in enumerate(table):
        if position + 1 == len(table):
            return entry.k
        following = table[position + 1]
        if entry.gap >= following.gap - following.standard_error:
            return entry.k
    return None
