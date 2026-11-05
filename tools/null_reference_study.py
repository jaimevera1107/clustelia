"""Paired study of the structureless reference families.

Mirrors the protocol of the November 2026 validation study: synthetic
datasets with 160 rows, k-means with ``k`` in 2..5, the full Clustelia
search (null screen, gap choice of ``k``, confirmation), and the same
seeds for every reference family so that the comparison is paired.

Scenarios
---------
- ``skewed_unimodal``: one lognormal cloud, no groups (every
  recommendation is spurious by design);
- ``gaussian_unimodal``: one correlated Gaussian cloud, no groups;
- ``blobs_separated``: three balanced, well separated blobs;
- ``blobs_unbalanced``: three blobs with sizes 100/40/20;
- ``blobs_overlapping``: three balanced blobs with wide spread.

Reported per scenario, family and scaling: recommendation rate with a
Wilson 95% interval, exact-k rate, mean ARI conditional on recommending
(with the count that supports it), boundary abstentions and mean fits.
The seeds are reused across conditions; runs are NOT independent across
conditions and the intervals describe one condition at a time.

Usage::

    python tools/null_reference_study.py --seeds 30 --out study.json
    python tools/null_reference_study.py --seeds 6 --quick --out pilot.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustelia import (
    AutoCluster,
    CandidateSpec,
    GeometrySpec,
    SearchBudget,
    SelectionPolicy,
    __version__,
)

N_ROWS: int = 160
K_RANGE: tuple[int, ...] = (2, 3, 4, 5)
SEED_OFFSET: int = 500_000
FAMILIES: tuple[str, ...] = ("pca_uniform", "gaussian_copula")
SCALINGS: tuple[str, ...] = ("standard", "none")
SCENARIOS: tuple[str, ...] = (
    "skewed_unimodal",
    "gaussian_unimodal",
    "blobs_separated",
    "blobs_unbalanced",
    "blobs_overlapping",
)
FULL_POLICY: dict[str, Any] = {
    "null_replicates": 19, "subsamples": 3, "seed_repeats": 2,
    "boundary_k_policy": "abstain",
}
QUICK_POLICY: dict[str, Any] = {
    "null_replicates": 9, "subsamples": 3, "seed_repeats": 2,
    "boundary_k_policy": "abstain",
}
MAX_FITS: int = 360


@dataclass(frozen=True)
class Dataset:
    """Synthetic dataset; ``y`` is ``None`` when there are no groups."""

    scenario: str
    seed: int
    X: np.ndarray
    y: np.ndarray | None
    true_k: int | None


def make_dataset(scenario: str, seed: int) -> Dataset:
    """Draw one dataset of ``scenario`` from ``seed``."""
    rng = np.random.default_rng(SEED_OFFSET + seed)
    state = int(rng.integers(2**31 - 1))
    if scenario == "skewed_unimodal":
        base = rng.standard_normal((N_ROWS, 3))
        return Dataset(scenario, seed, np.exp(base), None, None)
    if scenario == "gaussian_unimodal":
        cov = np.array([[1.0, 0.7, 0.3], [0.7, 1.0, 0.5], [0.3, 0.5, 1.0]])
        X = rng.multivariate_normal(np.zeros(3), cov, size=N_ROWS)
        return Dataset(scenario, seed, X, None, None)
    if scenario == "blobs_separated":
        X, y = make_blobs(N_ROWS, n_features=3, centers=3, cluster_std=0.5,
                          random_state=state)
        return Dataset(scenario, seed, X, y, 3)
    if scenario == "blobs_unbalanced":
        X, y = make_blobs([100, 40, 20], n_features=3, cluster_std=0.8,
                          random_state=state)
        return Dataset(scenario, seed, X, y, 3)
    if scenario == "blobs_overlapping":
        X, y = make_blobs(N_ROWS, n_features=3, centers=3, cluster_std=1.6,
                          random_state=state)
        return Dataset(scenario, seed, X, y, 3)
    raise ValueError(f"Unknown scenario {scenario!r}; expected {SCENARIOS}")


@dataclass(frozen=True)
class Outcome:
    """One search on one dataset under one condition."""

    scenario: str
    seed: int
    family: str
    scaling: str
    status: str
    recommended: bool
    selected_k: int | None
    at_boundary: bool
    true_k: int | None
    ari: float | None
    null_passed: bool | None
    fits_used: int
    seconds: float


def run_one(data: Dataset, family: str, scaling: str,
            policy_kwargs: dict[str, Any]) -> Outcome:
    """Fit the full Clustelia search and score it against the truth."""
    policy = SelectionPolicy(null_family=family, **policy_kwargs)
    started = time.perf_counter()
    result = AutoCluster(
        candidates=[CandidateSpec(algorithm="kmeans", k=k) for k in K_RANGE],
        geometry=GeometrySpec(scaling=scaling),
        selection=policy,
        budget=SearchBudget(max_fits=MAX_FITS),
        random_state=data.seed,
    ).fit(data.X).result_
    seconds = time.perf_counter() - started
    meta = result.metadata
    recommendation = result.recommendation
    ari = None
    if recommendation is not None and data.y is not None:
        labels = result.select(recommendation.candidate_id).labels_
        ari = float(adjusted_rand_score(data.y, labels))
    structure = meta.get("structure_test")
    return Outcome(
        scenario=data.scenario,
        seed=data.seed,
        family=family,
        scaling=scaling,
        status=result.selection_status,
        recommended=recommendation is not None,
        selected_k=meta.get("selected_k"),
        at_boundary=bool(meta.get("selected_k_at_boundary")),
        true_k=data.true_k,
        ari=ari,
        null_passed=(structure.get("passed")
                     if isinstance(structure, dict) else None),
        fits_used=int(meta["fits_used"]),
        seconds=round(seconds, 3),
    )


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float,
                                                                   float]:
    """Wilson score interval for a binomial proportion."""
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    half /= denominator
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def summarise(outcomes: Sequence[Outcome]) -> list[dict[str, Any]]:
    """Aggregate per scenario, family and scaling."""
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for family in FAMILIES:
            for scaling in SCALINGS:
                cell = [o for o in outcomes
                        if (o.scenario, o.family, o.scaling)
                        == (scenario, family, scaling)]
                if not cell:
                    continue
                n = len(cell)
                recommended = sum(o.recommended for o in cell)
                exact = sum(
                    o.recommended and o.selected_k == o.true_k for o in cell
                )
                aris = [o.ari for o in cell if o.ari is not None]
                low, high = wilson(recommended, n)
                rows.append({
                    "scenario": scenario,
                    "family": family,
                    "scaling": scaling,
                    "runs": n,
                    "recommended": recommended,
                    "recommendation_rate": recommended / n,
                    "wilson_95": [round(low, 3), round(high, 3)],
                    "exact_k": exact if cell[0].true_k else None,
                    "ari_mean_given_recommended": (
                        float(np.mean(aris)) if aris else None
                    ),
                    "ari_support": len(aris),
                    "boundary_abstentions": sum(
                        o.at_boundary and not o.recommended for o in cell
                    ),
                    "null_rejections": sum(
                        o.null_passed is False for o in cell
                    ),
                    "incomplete": sum(o.status == "incomplete" for o in cell),
                    "mean_fits": float(np.mean([o.fits_used for o in cell])),
                    "mean_seconds": float(np.mean([o.seconds for o in cell])),
                })
    return rows


def paired(outcomes: Sequence[Outcome]) -> list[dict[str, Any]]:
    """Paired ARI difference of the copula against the box, same seeds."""
    rows: list[dict[str, Any]] = []
    index = {(o.scenario, o.seed, o.family, o.scaling): o for o in outcomes}
    for scenario in SCENARIOS:
        for scaling in SCALINGS:
            deltas = []
            for key, box in index.items():
                if key[0] != scenario or key[2] != "pca_uniform" \
                        or key[3] != scaling:
                    continue
                copula = index.get((scenario, key[1], "gaussian_copula",
                                    scaling))
                if (copula is None or box.ari is None
                        or copula.ari is None):
                    continue
                deltas.append(copula.ari - box.ari)
            if deltas:
                rows.append({
                    "scenario": scenario,
                    "scaling": scaling,
                    "pairs_both_recommended": len(deltas),
                    "mean_ari_delta_copula_minus_box": float(np.mean(deltas)),
                })
    return rows


def print_table(rows: Sequence[dict[str, Any]]) -> None:
    """Human-readable summary on stdout."""
    print(f"{'scenario':20s} {'family':16s} {'scaling':9s} "
          f"{'rec':>7s} {'wilson95':>15s} {'k ok':>5s} {'ARI|rec':>8s} "
          f"{'n':>3s} {'bound':>5s} {'fits':>6s}")
    for r in rows:
        ari = (f"{r['ari_mean_given_recommended']:.3f}"
               if r["ari_mean_given_recommended"] is not None else "-")
        exact = "-" if r["exact_k"] is None else str(r["exact_k"])
        print(f"{r['scenario']:20s} {r['family']:16s} {r['scaling']:9s} "
              f"{r['recommended']:>3d}/{r['runs']:<3d} "
              f"[{r['wilson_95'][0]:.2f}, {r['wilson_95'][1]:.2f}] "
              f"{exact:>5s} {ari:>8s} {r['ari_support']:>3d} "
              f"{r['boundary_abstentions']:>5d} {r['mean_fits']:>6.0f}")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--scalings", nargs="+", default=list(SCALINGS))
    parser.add_argument("--quick", action="store_true",
                        help="9 null replicates instead of 19")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    policy_kwargs = QUICK_POLICY if args.quick else FULL_POLICY
    warnings.filterwarnings("ignore")
    outcomes: list[Outcome] = []
    for scenario in args.scenarios:
        for seed in range(args.seeds):
            data = make_dataset(scenario, seed)
            for scaling in args.scalings:
                for family in FAMILIES:
                    outcome = run_one(data, family, scaling, policy_kwargs)
                    outcomes.append(outcome)
                    print(f"{scenario:20s} seed={seed:<3d} {scaling:9s} "
                          f"{family:16s} {outcome.status:22s} "
                          f"k={outcome.selected_k} ari={outcome.ari}",
                          flush=True)
    summary = summarise(outcomes)
    print()
    print_table(summary)
    payload = {
        "package_version": __version__,
        "protocol": {
            "n_rows": N_ROWS, "k_range": list(K_RANGE),
            "max_fits": MAX_FITS, "seeds": args.seeds,
            "policy": policy_kwargs, "families": list(FAMILIES),
            "scalings": list(args.scalings),
        },
        "summary": summary,
        "paired": paired(outcomes),
        "outcomes": [asdict(o) for o in outcomes],
    }
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
