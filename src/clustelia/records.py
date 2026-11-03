"""Schema of the per-candidate record shown in ``results_``.

Records are plain dictionaries so that they map directly to the stable
columns of :meth:`ExperimentResult.candidates_table`. This module is the
single place that defines their keys and initial values.
"""

from __future__ import annotations

from typing import Any

from .config import CandidateSpec, candidate_parameters

__all__ = ["CandidateRow", "new_candidate_row"]

CandidateRow = dict[str, Any]


def new_candidate_row(
    spec: CandidateSpec, identifier: str, seed: int
) -> CandidateRow:
    """Return an initial, not-yet-fitted record for one candidate.

    Parameters
    ----------
    spec : CandidateSpec
        Candidate configuration.
    identifier : str
        Stable candidate identifier.
    seed : int
        Exploration seed.
    """
    return {
        "candidate_id": identifier,
        "algorithm": spec.algorithm,
        "parameters": candidate_parameters(spec),
        "status": "failed",
        "warnings": [],
        "error": None,
        "confirmed": False,
        "composition": None,
        "randomness": None,
        "rejection_reasons": [],
        "n_clusters": None,
        "coverage": None,
        "noise_count": None,
        "min_cluster_size": None,
        "max_cluster_size": None,
        "silhouette": None,
        "silhouette_reason": None,
        "metric_assigned_count": None,
        "converged": None,
        "primary_score": None,
        "dbcv": None,
        "dbcv_reason": None,
        "primary_unavailable": False,
        "heldout_loglikelihood": None,
        "validation_diagnostics": None,
        "fuzzy_diagnostics": None,
        "selection_metric": None,
        "seed": seed,
        "bic": None,
        "structure_test": None,
    }
