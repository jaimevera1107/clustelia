"""Capture or compare clustelia results for behavioural equivalence.

The 0.3.0.dev1 local references are captured in the same software
and numerical environment before and after the refactor. See
`docs/VALIDATION.md` for hashes, caveats and the 20/20 comparison.
Pickles depend on library versions: recapture in the same environment.
Never load untrusted pickle files.

Usage: python tools/equivalence.py capture OUT.pkl
       python tools/equivalence.py compare A.pkl B.pkl [--rtol TOL]

Timing columns are excluded. By default everything else must match
exactly, which is only meaningful when both captures come from the same
software and numerical environment. Across environments (other NumPy,
SciPy or pandas versions) last-bit floating-point differences appear
while statuses, recommendations and labels stay identical; use
``--rtol 1e-9`` there. With ``--rtol`` floats are compared with relative
and absolute tolerance ``TOL`` (NaN equals NaN); statuses,
recommendations, candidate identifiers, labels and every non-float value
must still match exactly.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_blobs

from clustelia import AutoCluster, GeometrySpec, Schema
from clustelia.config import SelectionPolicy

TIMING_TOKENS = ("second", "time", "elapsed", "duration")


def mixed_frame() -> pd.DataFrame:
    """Return the synthetic mixed-type table used by the Gower scenarios."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "firm_id": np.arange(120),
        "income": np.r_[rng.normal(100, 5, 60), rng.normal(300, 10, 60)],
        "activity": ["retail"] * 60 + ["industry"] * 60,
        "size": ["s", "m"] * 30 + ["m", "l"] * 30,
        "exports": [0, 1] * 30 + [1, 1] * 30,
    })


def scenarios() -> dict[str, tuple[dict[str, Any], Any]]:
    """Return the named scenarios as ``(AutoCluster kwargs, data)``."""
    X3, _ = make_blobs(n_samples=240, centers=3, cluster_std=0.4,
                       random_state=42)
    noise = np.random.default_rng(7).uniform(size=(200, 4))
    iris = load_iris().data
    frame = mixed_frame()
    schema = Schema(numeric=["income"], categorical=["activity"],
                    ordinal={"size": ("s", "m", "l")}, binary=["exports"],
                    identifiers=["firm_id"])
    gower = GeometrySpec(kind="gower", scaling="none")
    return {
        "numeric_default": ({"k_range": (2, 3, 4, 5)}, X3),
        "numeric_noise": ({"k_range": (2, 3, 4)}, noise),
        "iris_default": ({"k_range": (2, 3, 4, 5, 6)}, iris),
        "families": ({"families": ("kmeans", "minibatch", "birch",
                                   "bisecting", "spectral",
                                   "agglomerative", "medoids"),
                      "k_range": (2, 3, 4)}, X3),
        "gower_medoids": ({"schema": schema, "geometry": gower,
                           "families": ("medoids",),
                           "k_range": (2, 3, 4)}, frame),
        "gower_default": ({"schema": schema, "geometry": gower,
                           "k_range": (2, 3, 4)}, frame),
        "gmm_bic": ({"objective": "probabilistic",
                     "families": ("gmm", "bgmm"),
                     "k_range": (2, 3, 4)}, X3),
        "gmm_heldout": ({"objective": "probabilistic",
                         "probabilistic_metric": "heldout_loglikelihood",
                         "families": ("gmm",), "k_range": (2, 3, 4)}, X3),
        "density": ({"objective": "density"}, X3),
        "fuzzy": ({"objective": "fuzzy", "k_range": (2, 3, 4)}, X3),
    }


def policies() -> dict[str, SelectionPolicy | None]:
    """Return the selection policies each scenario is run under."""
    return {"default": None, "null0": SelectionPolicy(null_replicates=0)}


def run_one(kwargs: dict[str, Any], X: Any,
            policy: SelectionPolicy | None) -> dict[str, Any]:
    """Fit one scenario and snapshot its status, table and labels."""
    params = dict(kwargs, random_state=0)
    if policy is not None:
        params["selection"] = policy
    try:
        result = AutoCluster(**params).fit(X).result_
    except Exception as error:  # record failures as part of behaviour
        return {"error": f"{type(error).__name__}: {error}"}
    table = result.candidates_table()
    drop = [c for c in table.columns
            if any(token in c.lower() for token in TIMING_TOKENS)]
    table = table.drop(columns=drop)
    labels = {}
    for cid in table["candidate_id"]:
        try:
            labels[cid] = np.asarray(result.select(cid).labels_)
        except Exception as error:
            labels[cid] = f"{type(error).__name__}"
    rec = result.recommendation
    return {
        "status": result.selection_status,
        "recommendation": None if rec is None else rec.candidate_id,
        "table": table,
        "labels": labels,
        "dropped": drop,
    }


def capture(path: str) -> None:
    """Run every scenario and pickle the snapshots to ``path``."""
    out = {}
    for name, (kwargs, X) in scenarios().items():
        for pname, policy in policies().items():
            snap = run_one(kwargs, X, policy)
            out[(name, pname)] = snap
            print(f"{name:16s} {pname:8s}",
                  snap.get("status", snap.get("error")),
                  snap.get("recommendation", ""))
    with open(path, "wb") as handle:
        pickle.dump(out, handle)


def values_close(x: Any, y: Any, rtol: float) -> bool:
    """Whether two snapshot values match within ``rtol`` on floats.

    Floats (including NumPy floating scalars and arrays) are compared with
    relative and absolute tolerance ``rtol`` and NaN equals NaN. Mappings
    and sequences are compared recursively; every other value must be
    equal. Integers, booleans and strings are never given a tolerance.
    """
    float_types = (float, np.floating)
    if isinstance(x, float_types) and isinstance(y, float_types):
        return bool(np.isclose(x, y, rtol=rtol, atol=rtol, equal_nan=True))
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        ax, ay = np.asarray(x), np.asarray(y)
        if ax.shape != ay.shape:
            return False
        if ax.dtype.kind == "f" and ay.dtype.kind == "f":
            return bool(np.allclose(ax, ay, rtol=rtol, atol=rtol,
                                    equal_nan=True))
        return all(values_close(u, v, rtol)
                   for u, v in zip(ax.ravel().tolist(), ay.ravel().tolist(),
                                strict=True))
    if isinstance(x, Mapping) and isinstance(y, Mapping):
        return x.keys() == y.keys() and all(
            values_close(x[key], y[key], rtol) for key in x
        )
    if (isinstance(x, Sequence) and isinstance(y, Sequence)
            and not isinstance(x, (str, bytes))
            and not isinstance(y, (str, bytes))):
        return len(x) == len(y) and all(
            values_close(u, v, rtol) for u, v in zip(x, y, strict=True)
        )
    try:
        return bool(x == y) or (pd.isna(x) is True and pd.isna(y) is True)
    except (TypeError, ValueError):
        return False


def tables_close(x: pd.DataFrame, y: pd.DataFrame, rtol: float) -> bool:
    """Whether two candidate tables match cell by cell within ``rtol``."""
    if list(x.columns) != list(y.columns) or x.shape != y.shape:
        return False
    if not x.index.equals(y.index):
        return False
    return all(
        values_close(x.iloc[i, j], y.iloc[i, j], rtol)
        for i in range(x.shape[0]) for j in range(x.shape[1])
    )


def compare(path_a: str, path_b: str, rtol: float | None = None) -> int:
    """Compare two captures; return the number of differing runs.

    Parameters
    ----------
    path_a, path_b : str
        Pickles written by :func:`capture`.
    rtol : float or None
        ``None`` demands exact equality. A positive value compares floats
        with that relative and absolute tolerance (see
        :func:`values_close`).
    """
    with open(path_a, "rb") as handle:
        a = pickle.load(handle)
    with open(path_b, "rb") as handle:
        b = pickle.load(handle)
    failures = 0
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        problems = []
        if x is None or y is None:
            problems.append("missing")
        elif "error" in x or "error" in y:
            if x.get("error") != y.get("error"):
                problems.append("error differs")
        else:
            for field in ("status", "recommendation", "dropped"):
                if x[field] != y[field]:
                    problems.append(field)
            if rtol is None:
                try:
                    pd.testing.assert_frame_equal(
                        x["table"], y["table"], check_exact=True
                    )
                except AssertionError:
                    problems.append("table")
            elif not tables_close(x["table"], y["table"], rtol):
                problems.append("table")
            if x["labels"].keys() != y["labels"].keys():
                problems.append("label keys")
            else:
                for cid in x["labels"]:
                    lx, ly = x["labels"][cid], y["labels"][cid]
                    if isinstance(lx, str) or isinstance(ly, str):
                        same = lx == ly
                    else:
                        same = np.array_equal(lx, ly)
                    if not same:
                        problems.append(f"labels:{cid}")
        failures += bool(problems)
        print(f"{key[0]:16s} {key[1]:8s}",
              "IDENTICAL" if not problems else f"DIFF {problems}")
    mode = "exact" if rtol is None else f"rtol={rtol:g}"
    print(f"\n{len(a)} runs, {failures} differing ({mode})")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("out")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("a")
    compare_parser.add_argument("b")
    compare_parser.add_argument("--rtol", type=float, default=None)
    args = parser.parse_args(argv)
    if args.command == "capture":
        capture(args.out)
        return 0
    if args.rtol is not None and not args.rtol > 0:
        parser.error("--rtol must be positive")
    return 1 if compare(args.a, args.b, args.rtol) else 0


if __name__ == "__main__":
    sys.exit(main())
