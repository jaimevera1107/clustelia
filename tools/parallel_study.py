"""Measure the wall-clock effect of ``SearchBudget(workers=...)``.

Fits the same blobs with several worker counts and checks that every run
returns the sequential recommendation, fit count and number of clusters.
The first parallel fit of each worker count includes starting the worker
processes; later sizes reuse them, so the table shows both costs.

Usage::

    python tools/parallel_study.py
    python tools/parallel_study.py --sizes 500 2000 --workers 1 4 -1 \\
        --out parallel.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from sklearn.datasets import make_blobs

from clustelia import AutoCluster, SearchBudget


def measure(n: int, workers: int) -> dict[str, object]:
    """Fit ``n`` blobs with ``workers`` and return time and decisions."""
    X, _ = make_blobs(n, centers=4, cluster_std=1.0, random_state=0)
    started = time.perf_counter()
    result = AutoCluster(
        random_state=0, budget=SearchBudget(workers=workers)
    ).fit(X).result_
    seconds = time.perf_counter() - started
    recommended = result.recommendation
    return {
        "n": n,
        "workers": workers,
        "seconds": round(seconds, 2),
        "status": result.selection_status,
        "recommendation": (None if recommended is None
                           else recommended.candidate_id),
        "fits": int(result.metadata["fits_used"]),
        "k": result.metadata["selected_k"],
        "workers_used": int(result.metadata["parallel_workers"]),
        "structure_test_s": result.metadata["structure_test_seconds"],
        "stage_s": result.metadata["stage_seconds"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", nargs="*", type=int,
                        default=[500, 1000, 2000, 5000])
    parser.add_argument("--workers", nargs="*", type=int,
                        default=[1, 2, 4, 8, -1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    print(f"CPUs: {os.cpu_count()}")
    rows: list[dict[str, object]] = []
    baseline: dict[int, tuple[object, object, object]] = {}
    for workers in args.workers:
        for n in args.sizes:
            row = measure(n, workers)
            decisions = (row["recommendation"], row["fits"], row["k"])
            baseline.setdefault(n, decisions)
            row["same_as_first"] = decisions == baseline[n]
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(rows, indent=2))
    return 0 if all(r["same_as_first"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
