"""Effect of seeded restarts on alternating medoids (development data only).

For each dataset the medoids are fitted with the true number of groups on
the reference distance matrix (standardised euclidean or Gower), with
``n_init`` 1 and 5 and five base seeds. Reported per setting: total cost
(lower is better), ARI against the truth and the spread of ARI across
seeds (lower means less dependence on the initial draw).

Usage::

    python tools/medoid_restarts_study.py --out medoids.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import benchmark
import k_selection_study
import numpy as np
import openml_datasets
import pandas as pd
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

from clustelia.geometry import GowerMetric
from clustelia.medoids import AlternateMedoids

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
SETTINGS: tuple[int, ...] = (1, 5)
MAX_ROWS: int = 800


def _gower_matrix(frame: pd.DataFrame, schema: Any = None) -> np.ndarray:
    if schema is None:
        numeric = [c for c in frame.columns
                   if pd.api.types.is_numeric_dtype(frame[c])]
        categorical = [c for c in frame.columns if c not in numeric]
        metric = GowerMetric(numeric, categorical)
    else:
        metric = GowerMetric(schema.numeric, schema.categorical,
                             dict(schema.ordinal), schema.binary)
    constant = [c for c in metric.columns if frame[c].nunique() < 2]
    for group in (metric.numeric, metric.categorical, metric.binary):
        for column in constant:
            if column in group:
                group.remove(column)
    metric.fit(frame)
    return metric.distances(metric.transform(frame))


def datasets() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, distance matrix, truth) for the development data."""
    out = []
    for item in k_selection_study.build("development", 30):
        X = StandardScaler().fit_transform(item.X[:MAX_ROWS])
        out.append((f"synthetic:{item.name}", pairwise_distances(X),
                    item.y[:MAX_ROWS]))
    for scenario in benchmark.build_scenarios("development", 2):
        if scenario.name == "mixed_k3":
            out.append((f"benchmark:mixed_k3:{scenario.seed}",
                        _gower_matrix(scenario.data, scenario.schema),
                        scenario.truth))
    for name in ("zoo", "dermatology", "credit-approval", "glass", "ecoli"):
        X, y = openml_datasets.load(name)
        if all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
            values = StandardScaler().fit_transform(X.to_numpy(float))
            D = pairwise_distances(values)
        else:
            D = _gower_matrix(X)
        out.append((f"openml:{name}", D, y))
    return out


def evaluate(D: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Cost and ARI per setting over the base seeds."""
    k = int(len(np.unique(y)))
    record: dict[str, Any] = {"k": k}
    for n_init in SETTINGS:
        costs, aris = [], []
        for seed in SEEDS:
            model = AlternateMedoids(k, seed, n_init=n_init)
            labels = model.fit_predict(D)
            costs.append(model.inertia_)
            aris.append(adjusted_rand_score(y, labels))
        record[f"n{n_init}"] = {
            "mean_cost": float(np.mean(costs)),
            "mean_ari": float(np.mean(aris)),
            "sd_ari": float(np.std(aris)),
        }
    return record


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    results = {}
    for name, D, y in datasets():
        results[name] = evaluate(D, y)
        print(name, json.dumps(results[name]), flush=True)
        if args.out is not None:
            args.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
