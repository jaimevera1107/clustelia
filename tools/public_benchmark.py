"""Benchmark of the selection policies on public labelled datasets.

Class labels are used as a reference partition after selection, never
during it. They are not guaranteed to be the cluster structure of the
features: overlapping classes can make a lower ARI the honest answer.
Each dataset is also clustered by k-means with the true number of classes
(``oracle_kmeans_ari``) to show what a method that is *told* k reaches.

Two sources are available: ``sklearn`` (bundled datasets, no network) and
``openml`` (the frozen registry of ``tools/openml_datasets.py``, mixed
tables included, split into development and reserved datasets). For mixed
tables the oracle is Gower k-prototypes told the true number of classes.

Usage::

    python tools/public_benchmark.py --out public.json
    python tools/public_benchmark.py --source openml --split development
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openml_datasets
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

from clustelia import (
    AutoCluster,
    CandidateSpec,
    SearchBudget,
    SelectionPolicy,
)

Loader = Callable[[], tuple[np.ndarray, np.ndarray]]

LOADERS: dict[str, Loader] = {
    "iris": lambda: load_iris(return_X_y=True),
    "wine": lambda: load_wine(return_X_y=True),
    "breast_cancer": lambda: load_breast_cancer(return_X_y=True),
    "digits": lambda: load_digits(return_X_y=True),
}
"""Public datasets with class labels, keyed by name."""

RANDOM_STATE: int = 0


@dataclass
class PublicOutcome:
    """Result of one policy on one public dataset."""

    dataset: str
    policy: str
    rows: int
    features: int
    classes: int
    status: str
    k_found: int | None
    algorithm: str | None
    ari: float | None
    nmi: float | None
    oracle_kmeans_ari: float
    warnings: list[str]
    fits: int
    seconds: float


def _is_numeric(X: Any) -> bool:
    if isinstance(X, pd.DataFrame):
        return all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)
    return True


def oracle_ari(X: Any, y: np.ndarray) -> float:
    """ARI of a method told the true number of classes.

    Numeric data: k-means on standardised features. Data with nominal
    columns: Gower k-prototypes through clustelia (same geometry as the
    search).
    """
    k = len(np.unique(y))
    if _is_numeric(X):
        values = StandardScaler().fit_transform(np.asarray(X, dtype=float))
        labels = KMeans(k, n_init=10,
                        random_state=RANDOM_STATE).fit_predict(values)
        return float(adjusted_rand_score(y, labels))
    result = AutoCluster(
        candidates=(CandidateSpec("kprototypes", k),),
        selection=SelectionPolicy.p0(), random_state=RANDOM_STATE,
    ).fit(X).result_
    candidate = result.candidates_table().iloc[0].candidate_id
    return float(adjusted_rand_score(y, result.select(candidate).labels_))


oracle_kmeans_ari = oracle_ari
"""Backward-compatible name."""


def run(
    name: str,
    policy_name: str,
    policy: SelectionPolicy,
    oracle: float,
    k_max: int,
    data: tuple[Any, np.ndarray] | None = None,
) -> PublicOutcome:
    """Fit one policy on one dataset and compare with the class labels."""
    X, y = data if data is not None else LOADERS[name]()
    candidates = 3 * (k_max - 1)
    budget = SearchBudget(
        max_fits=max(320, candidates * (policy.null_replicates + 1) + 80),
        max_candidates=max(40, candidates),
    )
    started = time.perf_counter()
    result = AutoCluster(selection=policy, k_range=range(2, k_max + 1),
                         budget=budget,
                         random_state=RANDOM_STATE).fit(X).result_
    seconds = time.perf_counter() - started
    solution = (
        result.select(result.recommendation.candidate_id)
        if result.recommendation is not None else None
    )
    labels = solution.labels_ if solution is not None else None
    return PublicOutcome(
        dataset=name,
        policy=policy_name,
        rows=int(X.shape[0]),
        features=int(X.shape[1]),
        classes=int(len(np.unique(y))),
        status=result.selection_status,
        k_found=(int(len(np.unique(labels[labels >= 0])))
                 if labels is not None else None),
        algorithm=(solution.pipeline_.spec.algorithm
                   if solution is not None else None),
        ari=(float(adjusted_rand_score(y, labels))
             if labels is not None else None),
        nmi=(float(normalized_mutual_info_score(y, labels))
             if labels is not None else None),
        oracle_kmeans_ari=oracle,
        warnings=list(result.metadata.get("selection_warnings", [])),
        fits=int(result.metadata["fits_used"]),
        seconds=seconds,
    )


def policies() -> dict[str, SelectionPolicy]:
    """Policies compared by default."""
    return {
        "P0": SelectionPolicy.p0(),
        "P1": SelectionPolicy(k_selection="silhouette", version="P1-alpha1"),
        "P2": SelectionPolicy(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=("sklearn", "openml"),
                        default="sklearn")
    parser.add_argument("--split", choices=("development", "reserved"),
                        default="development",
                        help="OpenML split; reserved runs once.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--policies", nargs="*", default=None)
    parser.add_argument("--k-max", type=int, default=8,
                        help="Largest k of the automatic grid.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.source == "openml":
        names = args.datasets or openml_datasets.names(args.split)
        load = openml_datasets.load
    else:
        names = args.datasets or list(LOADERS)
        load = lambda name: LOADERS[name]()  # noqa: E731
    selected = policies()
    if args.policies:
        selected = {k: v for k, v in selected.items() if k in args.policies}
    outcomes: list[dict[str, Any]] = []
    for name in names:
        data = load(name)
        oracle = oracle_ari(*data)
        for policy_name, policy in selected.items():
            outcome = asdict(run(name, policy_name, policy, oracle,
                                 args.k_max, data=data))
            outcomes.append(outcome)
            print(json.dumps(outcome), flush=True)
            if args.out is not None:
                args.out.write_text(json.dumps(outcomes, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
