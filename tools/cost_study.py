"""Cost of a default search by number of rows, geometry and policy.

Each configuration runs in its own subprocess so that the peak resident
memory (``ru_maxrss``) belongs to that run alone; a hard wall-clock limit
stops runs that do not finish. Data: four Gaussian blobs in 10 dimensions
(euclidean) or three mixed-type groups (Gower), ``random_state=0``.

Usage::

    python tools/cost_study.py --rows 1000 10000 --out cost.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

POLICIES: dict[str, str] = {
    "P0": "SelectionPolicy.p0()",
    "P2": "SelectionPolicy()",
    "P2_B4": "SelectionPolicy(null_replicates=4)",
}

CHILD = r'''
import json, resource, sys, time
import numpy as np, pandas as pd
from sklearn.datasets import make_blobs
from clustelia import AutoCluster, GeometrySpec, SearchBudget, SelectionPolicy
rows, geometry, policy_name = int(sys.argv[1]), sys.argv[2], sys.argv[3]
policy = eval(sys.argv[4])
if geometry == "euclidean":
    X, _ = make_blobs(rows, n_features=10, centers=4, random_state=0)
    kwargs = {}
else:
    rng = np.random.default_rng(0)
    g = rng.integers(0, 3, rows)
    X = pd.DataFrame({
        "x": rng.normal(g * 4.0, 1.0), "y": rng.normal(-g * 2.0, 1.0),
        "colour": np.array(["red", "green", "blue"])[g],
        "shape": np.where(rng.random(rows) < 0.8,
                          np.array(["a", "b", "c"])[g], "a"),
    })
    kwargs = {"geometry": GeometrySpec(kind="gower", scaling="none")}
start = time.perf_counter()
result = AutoCluster(selection=policy, k_range=range(2, 9), random_state=0,
                     **kwargs).fit(X).result_
table = result.candidates_table()
print(json.dumps({
    "rows": rows, "geometry": geometry, "policy": policy_name,
    "status": result.selection_status,
    "seconds": time.perf_counter() - start,
    "fits": int(result.metadata["fits_used"]),
    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "errors": sorted({str(e)[:60] for e in table["error"].dropna()}),
    "fitted_families": sorted(set(table.loc[table.status == "fitted",
                                            "algorithm"])),
}))
'''


def run_one(rows: int, geometry: str, policy: str,
            limit_s: int) -> dict[str, Any]:
    """Run one configuration in a fresh interpreter."""
    try:
        done = subprocess.run(
            [sys.executable, "-c", CHILD, str(rows), geometry, policy,
             POLICIES[policy]],
            capture_output=True, text=True, timeout=limit_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"rows": rows, "geometry": geometry, "policy": policy,
                "status": f"wall limit {limit_s}s"}
    if done.returncode != 0:
        return {"rows": rows, "geometry": geometry, "policy": policy,
                "status": "crash", "stderr": done.stderr[-400:]}
    return json.loads(done.stdout.strip().splitlines()[-1])


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=int, nargs="+",
                        default=[1000, 10000, 50000])
    parser.add_argument("--geometries", nargs="+",
                        default=["euclidean", "gower"])
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    results = []
    for rows in args.rows:
        for geometry in args.geometries:
            for policy in args.policies:
                outcome = run_one(rows, geometry, policy, args.limit)
                results.append(outcome)
                print(json.dumps(outcome), flush=True)
                if args.out is not None:
                    args.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
