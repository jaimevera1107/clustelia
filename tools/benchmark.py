"""Reproducible synthetic benchmark of the selection policies.

The benchmark answers two questions with data whose truth is known:

1. On structureless data (one uniform or Gaussian cloud, independent mixed
   variables), how often does a policy still recommend a partition?
2. On data with clusters, how often does it recommend one, and how close
   is the recommendation to the truth (ARI, exact number of clusters)?

Scenarios are drawn from two disjoint seed ranges. Thresholds may only be
tuned on the ``development`` split; the ``reserved`` split is evaluated
once with the chosen configuration so that its numbers are not optimised.

Usage::

    python tools/benchmark.py --split development --repeats 3
    python tools/benchmark.py --split reserved --repeats 3 --out results.json

Each run is deterministic for a given split, repeat count and package
version.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clustelia import AutoCluster, Schema, SelectionPolicy

SPLIT_SEED_OFFSETS: dict[str, int] = {
    "development": 0,
    "reserved": 100_000,
    "reserved2": 200_000,
}
"""Disjoint seed ranges. ``reserved`` was spent on P1 (0.2.0a1) and then
inspected; ``reserved2`` is kept for the first evaluation of P2."""

N_ROWS: int = 300
"""Rows of every synthetic dataset."""

MIXED_LEVELS: tuple[str, ...] = ("low", "mid", "high")


@dataclass(frozen=True)
class Scenario:
    """One synthetic dataset and its truth.

    Attributes
    ----------
    name : str
        Scenario family.
    seed : int
        Seed that generated the data.
    data : numpy.ndarray or pandas.DataFrame
        Input table.
    truth : numpy.ndarray or None
        True labels; ``None`` for structureless data.
    schema : Schema or None
        Column roles for mixed data.
    """

    name: str
    seed: int
    data: Any
    truth: np.ndarray | None
    schema: Schema | None = None

    @property
    def has_structure(self) -> bool:
        """Whether the scenario contains clusters."""
        return self.truth is not None


ScenarioFactory = Callable[[np.random.Generator, int], Scenario]


def _blobs(rng: np.random.Generator, seed: int) -> Scenario:
    """Isotropic blobs with random count, spread and dimension."""
    centers = int(rng.choice((2, 3, 4, 6)))
    spread = float(rng.choice((0.5, 1.0, 1.5)))
    dimension = int(rng.choice((2, 5)))
    X, y = make_blobs(N_ROWS, n_features=dimension, centers=centers,
                      cluster_std=spread, random_state=seed)
    return Scenario(f"blobs_k{centers}_sd{spread}_d{dimension}", seed, X, y)


def _anisotropic(rng: np.random.Generator, seed: int) -> Scenario:
    """Blobs sheared by a random linear map."""
    X, y = make_blobs(N_ROWS, centers=3, cluster_std=0.8, random_state=seed)
    shear = np.array([[1.0, 0.0], [float(rng.uniform(-0.8, 0.8)), 1.0]])
    return Scenario("anisotropic_k3", seed, X @ shear, y)


def _mixed(rng: np.random.Generator, seed: int) -> Scenario:
    """Three mixed-type groups with 15% label noise per nominal variable."""
    groups = np.repeat(np.arange(3), N_ROWS // 3)
    flip = 0.15

    def noisy(values: np.ndarray, pool: Sequence[Any]) -> np.ndarray:
        """Replace a random 15% of ``values`` with draws from ``pool``."""
        mask = rng.random(len(values)) < flip
        values = values.astype(object)
        values[mask] = rng.choice(np.asarray(pool, dtype=object), mask.sum())
        return values

    frame = pd.DataFrame({
        "x": rng.normal(groups * 4.0, 1.0),
        "colour": noisy(np.array(["red", "green", "blue"])[groups],
                        ["red", "green", "blue"]),
        "level": noisy(np.array(MIXED_LEVELS)[groups], MIXED_LEVELS),
        "flag": noisy(np.array([0, 1, 1])[groups], [0, 1]).astype(int),
    })
    schema = Schema(numeric=["x"], categorical=["colour"],
                    ordinal={"level": MIXED_LEVELS}, binary=["flag"])
    return Scenario("mixed_k3", seed, frame, groups, schema)


def _uniform(rng: np.random.Generator, seed: int) -> Scenario:
    """Uniform hypercube (no clusters)."""
    dimension = int(rng.choice((2, 4)))
    return Scenario(f"null_uniform_d{dimension}", seed,
                    rng.uniform(size=(N_ROWS, dimension)), None)


def _gaussian(rng: np.random.Generator, seed: int) -> Scenario:
    """One correlated Gaussian cloud (no clusters)."""
    dimension = int(rng.choice((2, 5)))
    mixing = rng.normal(size=(dimension, dimension))
    data = rng.normal(size=(N_ROWS, dimension)) @ mixing
    return Scenario(f"null_gaussian_d{dimension}", seed, data, None)


def _mixed_null(rng: np.random.Generator, seed: int) -> Scenario:
    """Independent mixed variables (no clusters)."""
    frame = pd.DataFrame({
        "x": rng.normal(size=N_ROWS),
        "colour": rng.choice(["red", "green", "blue"], N_ROWS),
        "level": rng.choice(MIXED_LEVELS, N_ROWS),
        "flag": rng.integers(0, 2, N_ROWS),
    })
    schema = Schema(numeric=["x"], categorical=["colour"],
                    ordinal={"level": MIXED_LEVELS}, binary=["flag"])
    return Scenario("null_mixed", seed, frame, None, schema)


FACTORIES: tuple[ScenarioFactory, ...] = (
    _blobs, _blobs, _anisotropic, _mixed, _uniform, _gaussian, _mixed_null,
)
"""Scenario families; blobs appear twice to cover more configurations."""


def build_scenarios(split: str, repeats: int) -> list[Scenario]:
    """Return the deterministic scenarios of one split.

    Raises
    ------
    ValueError
        If ``split`` is unknown or ``repeats`` is not positive.
    """
    if split not in SPLIT_SEED_OFFSETS:
        raise ValueError(f"split must be one of {list(SPLIT_SEED_OFFSETS)}.")
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    scenarios = []
    offset = SPLIT_SEED_OFFSETS[split]
    for repeat in range(repeats):
        for position, factory in enumerate(FACTORIES):
            seed = offset + repeat * len(FACTORIES) + position
            scenarios.append(factory(np.random.default_rng(seed), seed))
    return scenarios


@dataclass
class Outcome:
    """Result of one policy on one scenario."""

    scenario: str
    seed: int
    has_structure: bool
    status: str
    recommended: bool
    ari: float | None
    k_found: int | None
    k_true: int | None
    fits: int
    seconds: float


@dataclass
class Summary:
    """Aggregate metrics of one policy on one split."""

    policy: str
    null_runs: int
    false_recommendation_rate: float
    structured_runs: int
    recommendation_rate: float
    mean_ari_when_recommended: float | None
    exact_k_rate_when_recommended: float | None
    utility: float
    mean_fits: float
    per_scenario: dict[str, dict[str, float]] = field(default_factory=dict)


def run_policy(scenario: Scenario, policy: SelectionPolicy) -> Outcome:
    """Fit ``AutoCluster`` with ``policy`` on one scenario."""
    started = time.perf_counter()
    model = AutoCluster(schema=scenario.schema, selection=policy,
                        random_state=scenario.seed)
    result = model.fit(scenario.data).result_
    labels = (
        result.select(result.recommendation.candidate_id).labels_
        if result.recommendation is not None else None
    )
    truth = scenario.truth
    return Outcome(
        scenario=scenario.name,
        seed=scenario.seed,
        has_structure=scenario.has_structure,
        status=result.selection_status,
        recommended=labels is not None,
        ari=(float(adjusted_rand_score(truth, labels))
             if labels is not None and truth is not None else None),
        k_found=int(len(np.unique(labels))) if labels is not None else None,
        k_true=int(len(np.unique(truth))) if truth is not None else None,
        fits=int(result.metadata["fits_used"]),
        seconds=time.perf_counter() - started,
    )


def summarise(name: str, outcomes: Sequence[Outcome]) -> Summary:
    """Aggregate outcomes into the benchmark metrics.

    The utility rewards recovery and penalises false recommendations:
    mean over structured runs of ``ARI`` (0 when abstaining) minus the
    false-recommendation rate on structureless runs.
    """
    null = [o for o in outcomes if not o.has_structure]
    structured = [o for o in outcomes if o.has_structure]
    recommended = [o for o in structured if o.recommended]
    false_rate = float(np.mean([o.recommended for o in null])) if null else 0
    recovery = float(np.mean([o.ari or 0.0 for o in structured]))
    per_scenario: dict[str, dict[str, float]] = {}
    for family in sorted({o.scenario.split("_d")[0] for o in outcomes}):
        members = [o for o in outcomes if o.scenario.startswith(family)]
        per_scenario[family] = {
            "runs": len(members),
            "recommended": float(np.mean([o.recommended for o in members])),
        }
    return Summary(
        policy=name,
        null_runs=len(null),
        false_recommendation_rate=false_rate,
        structured_runs=len(structured),
        recommendation_rate=(
            len(recommended) / len(structured) if structured else 0.0
        ),
        mean_ari_when_recommended=(
            float(np.mean([o.ari for o in recommended]))
            if recommended else None
        ),
        exact_k_rate_when_recommended=(
            float(np.mean([o.k_found == o.k_true for o in recommended]))
            if recommended else None
        ),
        utility=recovery - false_rate,
        mean_fits=float(np.mean([o.fits for o in outcomes])),
        per_scenario=per_scenario,
    )


def candidate_policies() -> dict[str, SelectionPolicy]:
    """Policies compared by default."""
    return {
        "P0": SelectionPolicy.p0(),
        "P1": SelectionPolicy(k_selection="silhouette", version="P1-alpha1"),
        "P2": SelectionPolicy(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=sorted(SPLIT_SEED_OFFSETS),
                        default="development")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--policies", nargs="*", default=None,
                        help="Subset of policy names to run.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    scenarios = build_scenarios(args.split, args.repeats)
    policies = candidate_policies()
    if args.policies:
        policies = {k: v for k, v in policies.items() if k in args.policies}
    report: dict[str, Any] = {"split": args.split, "repeats": args.repeats,
                              "summaries": [], "outcomes": {}}
    for name, policy in policies.items():
        outcomes = [run_policy(s, policy) for s in scenarios]
        summary = summarise(name, outcomes)
        report["summaries"].append(asdict(summary))
        report["outcomes"][name] = [asdict(o) for o in outcomes]
        print(json.dumps({k: v for k, v in asdict(summary).items()
                          if k != "per_scenario"}), flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
