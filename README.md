# clustelia

Automatic clustering that is allowed to say "there are no clusters".
clustelia explores candidate algorithms under a fixed reference geometry,
keeps every fitted pipeline, confirms the preferred solution by subsampling
and reseeding, and only recommends it if it beats structureless reference
data. Numeric, nominal, ordinal, binary and mixed tables are supported.

**Status: alpha (0.3.0a1).** The engineering contracts (validation,
reproducibility, budgets, persistence, typing, tests) are production-grade.
The selection policy P2 has been measured on synthetic benchmarks and on
18 public OpenML datasets, each with a development and a reserved split
([docs/VALIDATION.md](docs/VALIDATION.md)). It is strong on numeric data;
on wide categorical tables it tends to abstain (see Known limits).

## Installation

Python 3.11 or newer.

```bash
python -m pip install clustelia          # core
python -m pip install "clustelia[all]"   # + hdbscan, scikit-fuzzy, plots, Excel
```

From a source checkout use `python -m pip install -e ".[dev,all]"`
(`.[dev]` adds build, twine, pytest, ruff and mypy; `.[viz]` and
`.[reports]` add matplotlib and openpyxl separately).

`.[density]` and `.[fuzzy]` install each extra separately. Optional
backends are pinned to exact versions because their results are part of the
documented methodology; a different installed version disables the
capability with an explicit message. HDBSCAN *as a model* always uses
scikit-learn; the `density` extra only provides the DBCV validity index.

## Accessible API

```python
from clustelia import cluster
from sklearn.datasets import load_iris

frame = load_iris(as_frame=True).frame.drop(columns=["target"])
result = cluster(frame, effort="balanced")  # quick / balanced / thorough
print(result.summary())
print(result.report())
print(result.ranking_table())
solution = result.recommended_solution()
if solution is not None:
    print(solution.profile_table())
    solution.plot_sizes()  # optional matplotlib
```

DataFrame columns with unique, complete ID-like names are excluded
conservatively unless an explicit `Schema` is provided. The expert
`AutoCluster` API and its default settings are unchanged. The opt-in
`GeometrySpec(..., gower_backend="vectorized")` compiles Gower distances
through SciPy kernels; default Gower remains reference-stable. See
[convenience API, effort presets and visualisation](docs/CONVENIENCE_API.md).

## Quick start

```python
from sklearn.datasets import make_blobs
from clustelia import AutoCluster

X, _ = make_blobs(n_samples=240, centers=3, cluster_std=0.3, random_state=42)
model = AutoCluster(k_range=(2, 3, 4, 5), random_state=42).fit(X)
result = model.result_

print(result.selection_status)      # recommended | ambiguous |
                                    # no_admissible_solution | incomplete
print(result.candidates_table())    # unrounded evidence per candidate

if result.recommendation is not None:
    solution = result.select(result.recommendation.candidate_id)
    labels = solution.labels_
    profiles = solution.describe()
    new_labels = model.predict(X[:5])   # assign rows without refitting
```

Always check `result.recommendation is not None` in production code.
`model.labels_`, `model.best_estimator_`, `predict` and `fit_predict` raise
`NoRecommendationError` (with the result attached) when there is no
confirmed recommendation; `hasattr(model, "labels_")` returns `False` in that
case. `result.select(candidate_id)` retrieves any fitted candidate without
refitting it; selecting manually does not make it validated.

## Expert mode: same engine

```python
from clustelia import AutoCluster, CandidateSpec, GeometrySpec, SearchBudget

model = AutoCluster(
    geometry=GeometrySpec(kind="euclidean", scaling="robust"),
    candidates=(
        CandidateSpec("kmeans", k=3),
        CandidateSpec("kmeans", k=3, pca_variance=0.95),
        CandidateSpec("agglomerative", k=3),
        CandidateSpec("medoids", k=3),
    ),
    budget=SearchBudget(max_fits=200, max_candidates=20, timeout_s=300),
    random_state=42,
)
plan = model.plan(X)   # validates and lists candidates; fits nothing
model.fit(X)
```

`families=(...)` builds an automatic grid over `k_range`; `candidates=(...)`
takes explicit specifications. They are mutually exclusive. The reference
geometry is fixed before comparing candidates: PCA only changes the fitting
space, and every silhouette is computed in the common reference.

## Mixed data

```python
from clustelia import AutoCluster, GeometrySpec, Schema

model = AutoCluster(
    schema=Schema(
        numeric=["age", "income"],
        categorical=["activity"],                       # nominal
        ordinal={"education": ("primary", "secondary", "tertiary")},
        binary=["exports"],                             # asymmetric 0/1
        identifiers=["firm_id"],
    ),
    geometry=GeometrySpec(kind="gower", scaling="none",
                          weights={"income": 2.0}),     # optional
    k_range=(2, 3, 4),
).fit(df)
```

Every column must be declared exactly once; without a schema, numeric
dtypes become numeric and everything else nominal (ordinal and binary
semantics are never inferred). Any non-numeric feature selects Gower by
default. Weighted Gower averages per-variable dissimilarities in [0, 1]:

- numeric: difference over the training range (new out-of-range values
  are clipped to 1);
- nominal: match/mismatch; unseen categories raise an error;
- ordinal: difference of level positions over `levels - 1` of the declared
  scale; undeclared levels raise an error;
- binary (asymmetric, Gower 1971): mismatch counts 1, joint presence 0,
  and a joint absence (0, 0) is left out of the comparison. Declare a
  symmetric binary variable as nominal instead.

Weights are relative and default to 1. Two Gower families are searched by
default:

- `medoids`: alternating local descent on the dense distance matrix (not
  PAM), five seeded restarts, memory O(n²);
- `kprototypes`: Gower k-prototypes, assignable and memory O(n·k).
  Prototypes take the median (numeric), the mode (nominal, binary) and the
  lower-median level (ordinal). Without asymmetric binaries each update is
  an exact coordinate minimiser, so the cost never increases. On
  nominal-only data it is exactly k-modes with simple matching. It is not
  Huang's k-prototypes (squared euclidean plus a free `gamma`).

`solution.describe()` reports numeric quartiles, nominal proportions, the
median ordinal level with proportions in declared order, and the share of
ones of each binary variable.

Missing values are rejected by default. Under Gower they can be handled by
pairwise deletion, which compares each pair of rows on the variables
observed in both (Gower, 1971):

```python
GeometrySpec(kind="gower", scaling="none", missing="pairwise")
```

Every row needs at least one observed feature; summaries in `describe()`
use observed values and report the missing fraction per variable. The
treatment assumes missingness unrelated to the clusters.

### Known limits

On tables with many nominal columns the gap statistic often has no
maximum and the search abstains even when a useful partition exists: on
OpenML, P2 abstained on dermatology, credit-approval, lymph and mushroom,
where the silhouette policy recommended partitions with ARI 0.14–0.53
against the class labels; on zoo it was the best policy (0.94). If abstention is
not acceptable there, `SelectionPolicy(k_selection="silhouette")` restores
the 0.2.0a1 choice of k; inspect the result before relying on it.

## Objectives

| Objective | Adapters | Primary metric |
|---|---|---|
| `compact` | KMeans, MiniBatchKMeans, BIRCH, BisectingKMeans, Spectral, Ward, medoids, Gower k-prototypes | Reference silhouette |
| `probabilistic` | GaussianMixture, BayesianGaussianMixture | BIC (GMM only) or reserved held-out likelihood |
| `density` | DBSCAN, OPTICS, HDBSCAN (scikit-learn) | DBCV in the euclidean reference (`density` extra) |
| `fuzzy` | Fuzzy C-Means (scikit-fuzzy) | Generalised Xie-Beni with fixed fuzzifier (`fuzzy` extra) |

Scores are never compared across objectives, geometries or representations;
runs mixing representations under BIC, likelihood or Xie-Beni are rejected.
When the density backend or a compatible geometry is missing, the search
returns `incomplete` with the reason and never substitutes another metric.
Ward, Spectral, DBSCAN, OPTICS and HDBSCAN are transductive and reject
`assign`. `adapter_capabilities()` lists capabilities and availability.

`solution.membership_` carries explicit semantics: `mixture_posterior`,
`fuzzy_membership` or `density_association_strength` (the latter is a
strength of the assigned cluster, not a distribution over clusters).
`solution.structure_` exposes the Ward tree or the OPTICS reachability.

## Selection policy

`SelectionPolicy()` (P2) runs three steps for the compact objective.

**1. Is there structure at all?** Stability cannot tell clusters from
noise: k-means splits a uniform square into stable pieces, and the
stability-only policy recommended a partition on about three quarters of
structureless benchmark datasets. Every explored candidate is refitted on
`null_replicates=9` structureless datasets: uniform in the principal-axis
box of the data (euclidean, as in the gap statistic) or independent column
permutations (Gower). The best silhouette on the data must beat the best
silhouette on every replicate, which bounds false recommendations by
`1 / (B + 1)` = 10% when the data follow the reference. If it fails, the
result is `no_admissible_solution` and no confirmation fit is spent.

**2. How many clusters?** The silhouette rewards the top level of a
hierarchy, so it merges nearby clusters. The number of clusters is chosen
with the gap statistic (Tibshirani, Walther and Hastie, 2001), computed
from the null fits already run in step 1: for each `k` the candidate with
the largest `Gap(k) = mean log W*_k - log W_k` represents that `k`, and
the smallest `k` with `Gap(k) >= Gap(k+1) - s_(k+1)` is selected. The gap
table and `selected_k` are in `result.metadata`. When the selected `k` is
the largest one explored, `metadata["selection_warnings"]` says so: widen
`k_range` (and `SearchBudget`) before trusting that number.

**3. Which candidate?** Only candidates with the selected `k` are
confirmed: 10 subsamples of 80% of rows and 3 extra seeds per stochastic
candidate, each replica refitting the whole pipeline. A candidate is
admissible when coverage ≥ 0.8, at least 80% of replicas are evaluable and
the median ARI is ≥ 0.7 on every applicable axis. Among admissible ones,
silhouettes within 0.02 of the best are tied and broken by stability,
coverage, cost (in fits) and identifier.

The null fits cost `null_replicates × candidates` for structured data
(early exit on structureless data), which is why the default `max_fits` is
320. Alternatives: `k_selection="silhouette"` keeps step 1 but chooses `k`
by the silhouette band (P1, 0.2.0a1); `null_scope="family"` refits only
the preferred family (cheaper, optimistic, implies the silhouette rule);
`SelectionPolicy.p0()` restores the stability-only policy of 0.1.0a3.
Probabilistic (BIC or held-out likelihood), density and fuzzy objectives
keep their own metrics and are not tested against a null reference.

Resampling assumes rows suitable for independent subsampling; time series,
panels and grouped data need a protocol that is not implemented yet.

## Execution contracts

- Input is copied, never modified. Missing values, sparse matrices and
  sample weights raise errors; nothing is imputed or densified silently.
- Assignment requires the training columns in the same order; one row is
  enough.
- `n_jobs` limits native threads per fit. The null-reference stage (most of
  the cost) refits candidates in worker processes: `SearchBudget(workers=None)`
  chooses automatically (1 below 1000 rows, else up to 8), `workers=1` runs
  sequentially and `workers=-1` uses every CPU. Results, seeds and fit counts
  do not depend on it; memory does (each worker holds a copy of the data).
- `fits_used` counts every attempt, failures and replicas included, and a
  confirmation reserve is set aside before exploring. A truncated grid or
  confirmation yields `incomplete`.
- `timeout_s` is a soft deadline checked between fits. `max_pairwise_mb`
  bounds a *preventive estimate* of dense pairwise matrices
  (`MemoryBudgetError`, a `MemoryError`), not the process RSS.
- Backend errors are recorded per candidate; if all fail,
  `SearchFailedError` is raised with the result attached.
  `error_policy="raise"` helps debugging. Programming errors propagate.
- All deliberate errors derive from `ClusteliaError`.
- The library logs through the `clustelia` logger (silent by default).

## Persistence

```python
from clustelia import save, load

save(model, "model.pkl")                      # atomic write
restored = load("model.pkl", trusted=True)
```

Files are pickles containing the pipelines and the training table,
identifiers included. **Loading a pickle can execute code:** only load files
you created. The environment check (package, Python series, NumPy, pandas,
SciPy, scikit-learn and the fuzzy backend) happens after deserialisation; it
is a compatibility check, not a security boundary. No cross-version
migration is offered.

## Development

```bash
python -m pip install -e ".[dev,all]"
python -m unittest discover -s tests -v      # or: pytest
ruff check src tests tools examples
python tools/check_source.py src/clustelia
mypy
python -m build && python -m twine check --strict dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the architecture and the release
checklist, [docs/METHODS.md](docs/METHODS.md) for methodological decisions
and [docs/VALIDATION.md](docs/VALIDATION.md) for the verified environment.
`docs/SPECIFICATION_v0_1.md` is the original design specification (Spanish).

## License

BSD 3-Clause, see `LICENSE`.


## Presets, Gower comparability, CLARA medoids and large tables

``clustelia.cluster`` provides presets and independent null-testing depth:

```python
from clustelia import cluster, load

result = cluster(X, effort="quick", null_replicates=19,
                 families=("kmeans",), k_range=(2, 3, 4))
print(result.summary())
result.export_report("report.html")
result.save("run.clustelia")  # trusted pickle, includes training data
restored = load("run.clustelia", trusted=True)
```

For Gower with missing values, pairs without comparable information now
**raise by default** instead of being treated as identical. To preserve a
usable finite matrix under frequent missingness, request an explicit
penalty, e.g.

```python
from clustelia import GeometrySpec
geometry = GeometrySpec(kind="gower", scaling="none", missing="pairwise",
                        incomparable="max", min_comparable_share=0.25,
                        null_blocks=(("dummy_a", "dummy_b"),))
```

``incomparable="zero"`` is a legacy compatibility escape hatch and must
not be interpreted as evidence of similarity. ``null_blocks`` preserve
within-block dependencies in Gower permutation-reference data.

**Large tables.** Ward and medoids build a dense n-by-n matrix, so above
about 3300 rows the default 256 MiB guard (`SearchBudget(max_pairwise_mb)`)
rejects them. The search then continues with the remaining families and
says so in `metadata["selection_warnings"]`; raise the limit or use CLARA
medoids (below). The null-reference stage, most of the cost, runs in worker
processes on tables of 1000 rows or more (`SearchBudget(workers=...)`).

CLARA-style medoids are available as an **opt-in** candidate that fits
sampled distance matrices and evaluates medoids on all training observations
using bounded row blocks. It is an approximation, not PAM:

```python
from clustelia import AutoCluster, CandidateSpec
experiment = AutoCluster(candidates=(
    CandidateSpec("clara", k=3, clara_sample_size=128, clara_samples=5),
), random_state=42).fit(X)
```

The separate common reference metrics and validation may still allocate
dense matrices on their bounded metric samples; CLARA does not make all
of Clustelia out-of-core. See `docs/DEV2_CHANGES.md` for contracts,
performance caveats, limitations and migration notes.
