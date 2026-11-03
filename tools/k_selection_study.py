"""Offline study of published rules for choosing the number of clusters.

Every rule sees the same k-means fits in the same standardised space, so
the comparison isolates the choice of k from the choice of family:

- gap statistic (Tibshirani, Walther and Hastie, 2001) with the PCA-box
  reference, and its published decision rules as implemented in R's
  ``cluster::maxSE``: ``Tibs2001SEmax`` (clustelia P2), ``firstSEmax``,
  ``globalSEmax`` (Dudoit and Fridlyand, 2002) and ``firstmax``;
- the same gap computed in the principal subspace that keeps 90% of the
  variance (``gap_pca``, Tibs2001SEmax rule);
- prediction strength (Tibshirani and Walther, 2005): the largest k whose
  mean two-fold prediction strength is at least 0.8.

Datasets are synthetic with known k, drawn from disjoint seed ranges for
development and ``reserved3``; the reserved split is run once.

Usage::

    python tools/k_selection_study.py --split development --out dev.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

SPLIT_SEED_OFFSETS: dict[str, int] = {"development": 300_000,
                                      "reserved3": 400_000}
K_MAX: int = 10
REPLICATES: int = 9
PS_THRESHOLD: float = 0.8
PS_REPEATS: int = 3
PCA_VARIANCE: float = 0.90
N_INIT: int = 4
RULES: tuple[str, ...] = ("Tibs2001SEmax", "firstSEmax", "globalSEmax",
                          "firstmax", "gap_pca", "prediction_strength")


@dataclass(frozen=True)
class Dataset:
    """Synthetic dataset with known number of clusters."""

    name: str
    seed: int
    X: np.ndarray
    y: np.ndarray


def make_dataset(seed: int) -> Dataset:
    """Draw one configuration; every factor is sampled from ``seed``."""
    rng = np.random.default_rng(seed)
    k = int(rng.choice((2, 3, 4, 5, 6, 8)))
    dim = int(rng.choice((2, 5, 10, 30, 64)))
    n = int(rng.choice((200, 400, 800)))
    spread = float(rng.choice((0.6, 1.0, 1.6)))
    imbalance = bool(rng.random() < 0.3)
    noise_dims = int(rng.choice((0, 0, 2, 10)))
    weights = (rng.dirichlet(np.full(k, 2.0)) if imbalance
               else np.full(k, 1 / k))
    sizes = np.maximum(10, np.round(weights * n)).astype(int)
    # Centres drawn in a fixed box: separation grows with the dimension.
    X, y = make_blobs(n_samples=sizes, n_features=dim, cluster_std=spread,
                      center_box=(-6.0, 6.0), random_state=seed)
    if noise_dims:
        scale = X.std(axis=0).mean()
        X = np.hstack([X, rng.normal(0, scale, size=(len(X), noise_dims))])
    name = (f"k{k}_d{dim}+{noise_dims}_n{len(X)}_sd{spread}"
            f"{'_imb' if imbalance else ''}")
    return Dataset(name, seed, X, y)


def build(split: str, count: int) -> list[Dataset]:
    """Deterministic datasets of one split."""
    offset = SPLIT_SEED_OFFSETS[split]
    return [make_dataset(offset + i) for i in range(count)]


def _log_w(X: np.ndarray, labels: np.ndarray) -> float:
    """log of pooled within-cluster dispersion (sum of squares / 2n_r)."""
    total = 0.0
    for cluster in np.unique(labels):
        members = X[labels == cluster]
        total += ((members - members.mean(axis=0)) ** 2).sum()
    return float(np.log(max(total, 1e-12)))


def _kmeans(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k == 1:
        return np.zeros(len(X), dtype=int)
    return KMeans(k, n_init=N_INIT, random_state=seed).fit_predict(X)


def _pca_box(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    centre = X.mean(axis=0)
    _, _, axes = np.linalg.svd(X - centre, full_matrices=False)
    scores = (X - centre) @ axes.T
    box = rng.uniform(scores.min(axis=0), scores.max(axis=0),
                      size=scores.shape)
    return box @ axes + centre


def gap_curve(X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray,
                                                 dict[int, np.ndarray]]:
    """Gap and standard error for k = 1..K_MAX, plus the data partitions."""
    rng = np.random.default_rng(seed)
    ks = range(1, K_MAX + 1)
    partitions = {k: _kmeans(X, k, seed) for k in ks}
    observed = np.array([_log_w(X, partitions[k]) for k in ks])
    null = np.empty((REPLICATES, K_MAX))
    for b in range(REPLICATES):
        Z = _pca_box(X, rng)
        null[b] = [_log_w(Z, _kmeans(Z, k, seed + b)) for k in ks]
    gap = null.mean(axis=0) - observed
    se = null.std(axis=0) * np.sqrt(1 + 1 / REPLICATES)
    return gap, se, partitions


def max_se(gap: np.ndarray, se: np.ndarray, method: str) -> int:
    """Port of R ``cluster::maxSE``; index 0 of ``gap`` is k=1.

    As in clustelia, the rule searches k >= 2 (structure versus none is the
    null-reference test's job), so the curve is truncated before applying
    R's logic and the answer shifted back.
    """
    return _max_se(gap[1:], se[1:], method) + 1


def _max_se(gap: np.ndarray, se: np.ndarray, method: str) -> int:
    """R ``cluster::maxSE`` on a curve whose index 0 is position 1."""
    K = len(gap)
    if method == "firstmax":
        decreasing = np.flatnonzero(np.diff(gap) <= 0)
        return int(decreasing[0] + 1) if len(decreasing) else K
    if method == "globalSEmax":
        top = int(np.argmax(gap))
        return int(np.flatnonzero(gap >= gap[top] - se[top])[0] + 1)
    if method == "firstSEmax":
        decreasing = np.flatnonzero(np.diff(gap) <= 0)
        first = int(decreasing[0]) if len(decreasing) else K - 1
        return int(np.flatnonzero(gap >= gap[first] - se[first])[0] + 1)
    if method == "Tibs2001SEmax":
        hits = np.flatnonzero(gap[:-1] >= gap[1:] - se[1:])
        return int(hits[0] + 1) if len(hits) else K
    raise ValueError(method)


def prediction_strength(X: np.ndarray, seed: int) -> int:
    """Largest k >= 2 with mean prediction strength >= PS_THRESHOLD.

    Falls back to 2 when no k passes (k=1 is the null test's job).
    """
    rng = np.random.default_rng(seed)
    chosen = 2
    for k in range(2, K_MAX + 1):
        values = []
        for repeat in range(PS_REPEATS):
            order = rng.permutation(len(X))
            test, train = order[: len(X) // 2], order[len(X) // 2:]
            km_train = KMeans(k, n_init=N_INIT,
                              random_state=seed + repeat).fit(X[train])
            test_labels = KMeans(k, n_init=N_INIT, random_state=seed + repeat
                                 ).fit_predict(X[test])
            predicted = km_train.predict(X[test])
            strengths = []
            for cluster in range(k):
                members = np.flatnonzero(test_labels == cluster)
                if len(members) < 2:
                    continue
                same = predicted[members][:, None] == predicted[members][None]
                pairs = len(members) * (len(members) - 1)
                strengths.append((same.sum() - len(members)) / pairs)
            values.append(min(strengths) if strengths else 0.0)
        if np.mean(values) >= PS_THRESHOLD:
            chosen = k
    return chosen


@dataclass
class RuleOutcome:
    """One dataset under every rule."""

    name: str
    seed: int
    k_true: int
    k_chosen: dict[str, int]
    ari: dict[str, float]
    oracle_ari: float


def evaluate(dataset: Dataset) -> RuleOutcome:
    """Apply every rule to one dataset."""
    X = StandardScaler().fit_transform(dataset.X)
    gap, se, partitions = gap_curve(X, dataset.seed)
    chosen = {m: max_se(gap, se, m) for m in RULES[:4]}
    reduced = PCA(PCA_VARIANCE, svd_solver="full").fit_transform(X)
    gap_r, se_r, _ = gap_curve(reduced, dataset.seed)
    chosen["gap_pca"] = max_se(gap_r, se_r, "Tibs2001SEmax")
    chosen["prediction_strength"] = prediction_strength(X, dataset.seed)
    k_true = int(len(np.unique(dataset.y)))
    ari = {rule: float(adjusted_rand_score(dataset.y, partitions[k]))
           for rule, k in chosen.items()}
    return RuleOutcome(dataset.name, dataset.seed, k_true, chosen, ari,
                       float(adjusted_rand_score(dataset.y,
                                                 partitions[k_true])))


def summarise(outcomes: Sequence[RuleOutcome]) -> dict[str, Any]:
    """Exact-k rate, mean ARI and over/under-estimation per rule."""
    summary: dict[str, Any] = {}
    for rule in RULES:
        k = np.array([o.k_chosen[rule] for o in outcomes])
        truth = np.array([o.k_true for o in outcomes])
        summary[rule] = {
            "exact_k": float(np.mean(k == truth)),
            "over": float(np.mean(k > truth)),
            "under": float(np.mean(k < truth)),
            "mean_ari": float(np.mean([o.ari[rule] for o in outcomes])),
        }
    summary["oracle_mean_ari"] = float(np.mean([o.oracle_ari
                                                for o in outcomes]))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=sorted(SPLIT_SEED_OFFSETS),
                        default="development")
    parser.add_argument("--count", type=int, default=90)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    outcomes = []
    for dataset in build(args.split, args.count):
        outcome = evaluate(dataset)
        outcomes.append(outcome)
        print(json.dumps(asdict(outcome)), flush=True)
        if args.out is not None:
            args.out.write_text(json.dumps(
                {"split": args.split, "summary": summarise(outcomes),
                 "outcomes": [asdict(o) for o in outcomes]}, indent=1))
    print(json.dumps(summarise(outcomes), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
