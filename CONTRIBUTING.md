# Contributing

## Architecture

Each module has one responsibility; `AutoCluster` is a thin facade.

| Module | Responsibility |
|---|---|
| `config` | Immutable, validated specifications (no fitted state) |
| `schema` | Input validation, schema inference, row identity |
| `geometry` | Reference geometry (euclidean/Gower strategies), memory guard |
| `medoids`, `prototypes`, `fuzzy` | Estimators implemented or wrapped by clustelia |
| `adapters` | Capability registry, estimator factories, fitted pipeline |
| `planning` | Candidate grid, capability exclusions, confirmation reserve |
| `execution` | Budgeted fits: counting, deadline, seeds, warnings, threads |
| `scoring` | Objective-specific primary scores (strategy per objective) |
| `evaluation` | Partition metrics and stability comparisons |
| `confirmation` | Composition and seed replicas of the shortlist |
| `selection` | Admissibility, shortlist, Pareto frontier, preference |
| `structure` | Null-reference generation and test of the recommendation |
| `records` | Single definition of the per-candidate record schema |
| `results` | `ExperimentResult`, `Solution`, external evaluation |
| `search` | `AutoCluster` and the orchestration of one run |
| `persistence` | Atomic save and environment-checked load |
| `density`, `optional` | DBCV and pinned optional backends |

Only names exported by `clustelia/__init__.py` are public.

## Rules

- PEP 8 with 79-character lines, complete type annotations, and NumPy-style
  docstrings on every public module, class and function
  (`python tools/check_source.py src/clustelia` enforces this).
- No hidden global state: every random draw uses a seed derived with
  `stable_seed`; never call NumPy's global RNG.
- Behaviour changes need a test and a CHANGELOG entry. Changes to the
  selection methodology also need a new `SelectionPolicy.version` and a
  run of `tools/benchmark.py` on the development split; the reserved split
  is run once per methodology version.

## Behavioural equivalence

Refactors must keep results identical (compare with
`SelectionPolicy(null_replicates=0)` against releases before 0.2.0a1). Compare candidate tables, statuses,
recommendations and labels on representative scenarios before and after the
change (timing columns excluded).

`tools/equivalence.py compare A.pkl B.pkl` is exact and only meaningful
when both captures come from the same software and numerical environment.
Across environments last-bit float differences appear (statuses,
recommendations and labels stay identical): use `--rtol 1e-9`, and
recapture the reference in your own environment when a strict check is
needed.

## Local verification

`python tools/verify_all.py --install` runs the CI checks locally and writes
a report; see `docs/LOCAL_VERIFICATION.md`.

## Release checklist

1. License: BSD-3-Clause (set); keep `LICENSE` and `pyproject.toml` in sync.
2. Confirm the distribution name `clustelia` is registered on PyPI.
3. Update `src/clustelia/_version.py` and `CHANGELOG.md`.
4. Make sure CI is green on every OS / Python / extras combination, and
   that the `typecheck` job passes (then drop its `continue-on-error`).
5. Configure a PyPI trusted publisher for this repository and a `pypi`
   environment, then push a tag `vX.Y.Z` matching the version.
