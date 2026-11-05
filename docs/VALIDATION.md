> **Archive note (pre-publication cleanup):** The original validation
> records below are preserved as historical evidence, not re-run claims
> for this cleaned package. The public repository retains the frozen
> `reference_0.3.0a1.pkl`; previous `.pkl` snapshots, including their
> duplicate filenames, are in the companion archive
> `clustelia_referencias_historicas.zip`. The historic commands shown
> below refer to the original directory before this cleanup.
> See [equivalence/README.md](equivalence/README.md).

## Unreleased — null family and boundary policy (5 November 2026 UTC)

Branch `feat/null-family-and-boundary-policy`, Linux, Python 3.11,
NumPy/SciPy/scikit-learn as pinned by `.[all,dev]`.

### Engineering checks

- `pytest`: 196 passed, 39 subtests passed (182 on `main`; 14 new in
  `tests/test_null_family.py`).
- `ruff check src tests tools`: clean. `mypy`: clean, 28 source files.
- Behavioural equivalence (`tools/equivalence.py`, exact mode): a capture
  of this branch against a capture of `main` taken in the same
  environment is **20/20 IDENTICAL**. With the defaults unchanged
  (`null_family="auto"`, `boundary_k_policy="warn"`) no status,
  recommendation, candidate table or label moves.
- Against the frozen `reference_0.3.0a1.pkl` the same capture differs in
  10/20 runs at `--rtol 1e-9`; `main` differs in exactly the same 10
  runs, so the differences are environmental (density/fuzzy backends and
  last-bit numerics of this machine), not introduced here. The reference
  is not regenerated, per the traceability decision of the November 2026
  study.

### Null-reference study (`tools/null_reference_study.py`, 30 seeds)

Protocol: n = 160, k-means only, k in 2..5, B = 19 replicates,
`boundary_k_policy="abstain"`, max 360 fits, the same 30 seeds in every
condition. Five scenarios × 2 families × 2 scalings = 600 searches.
Raw outcomes and summaries: `docs/benchmark/null_reference_study_30.json`.
Development data: the scenarios were chosen knowing the failure mode under
study; this is evidence for a decision, not an out-of-sample validation.

| Scenario | Scaling | `pca_uniform` rec. | `gaussian_copula` rec. | ARI given rec. (box / copula) |
|:--|:--|--:|--:|:--|
| Skewed unimodal (no groups) | standard | 17/30 | 0/30 | – |
| Skewed unimodal (no groups) | none | 20/30 | 1/30 | – |
| Gaussian unimodal (no groups) | standard | 0/30 | 2/30 | – |
| Gaussian unimodal (no groups) | none | 0/30 | 2/30 | – |
| 3 separated blobs | standard | 29/30 | 30/30 | 0.980 / 0.981 |
| 3 separated blobs | none | 30/30 | 30/30 | 0.997 / 0.997 |
| 3 unbalanced blobs (100/40/20) | standard | 29/30 | 30/30 | 0.872 / 0.908 |
| 3 unbalanced blobs (100/40/20) | none | 30/30 | 30/30 | 0.989 / 0.970 |
| 3 overlapping blobs | standard | 30/30 | 29/30 | 0.924 / 0.934 |
| 3 overlapping blobs | none | 30/30 | 29/30 | 0.947 / 0.912 |

Reading:

- The copula removes the skewed-marginal false recommendations of the box
  reference (Wilson 95% upper bound 11% at 0/30) while keeping the
  sensitivity of every structured scenario within one run of the box.
  Paired ARI differences on seeds where both recommend are within
  ±0.05.
- On the correlated Gaussian cloud the copula recommended 2/30 (6.7%,
  interval 2–21%), the box 0/30. Both are compatible with the nominal
  1/(B + 1) = 5% resolution; neither reference is exchangeable with the
  data, so this is not a calibrated error rate.
- This result differs from the November 2026 study, where the copula
  variant then under test detected only 3/30 unbalanced mixtures. That
  variant lived on a local branch with its own generator and dataset
  design; the two are not the same implementation and the numbers are
  not comparable. The present generator maps normal scores back to the
  observed values by rank, so every marginal is preserved exactly.
- Boundary abstentions (`boundary_k_policy="abstain"`) occurred in 5 of
  600 searches, all with the box reference at `standard` scaling.

Decision: `pca_uniform` stays the default and `gaussian_copula` stays
opt-in. Promotion requires a reserved split not used to design the
scenarios, more dimensions and families, and real labelled data.

## 0.3.0.dev2 — local full-suite run (2 November 2026 UTC)

After the new contract tests and legacy compatibility tests were updated,
`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -q` completed successfully:

```text
Ran 147 tests in 98.112s
OK (skipped=10)
```

The ten skips concern optional dependencies and/or tests that cannot run
in this local environment. The successful run is not evidence of green
Windows/macOS CI, Ruff/Mypy, or all optional backends. Local source checking
(`python tools/check_source.py src/clustelia tests tools examples`) reported
zero issues; an offline wheel was installed into a separate target and
imported successfully. Release CI must rebuild using `python -m build`.

# 0.3.0.dev2 checkpoint notice

`0.3.0.dev2` retains exact candidate identifiers, fitted labels,
selection decisions, and per-candidate records for all 20 historical
`tools/equivalence.py` scenarios. The fresh reference
`docs/equivalence/reference_0.3.0.dev2.pkl` was captured locally on
Python 3.13 / Linux with NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3,
and scikit-learn 1.8.0. Then:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python tools/equivalence.py compare \
  docs/equivalence/reference_0.3.0.dev1.pkl \
  docs/equivalence/reference_0.3.0.dev2.pkl
```

**Result: 20/20 IDENTICAL, 0 differences**, without tolerances.
The historical baseline has not been overwritten. This does not imply
equivalence for newly introduced edge cases (exact zero Gap, noncomparable
Gower pairs), opt-in CLARA, or other environments and optional backends.
See `DEV2_CHANGES.md` for the intentional changes in those areas.

**Windows 11 / Python 3.14.6 (3 November 2026, `tools/verify_all.py
--install`).** NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.0, scikit-learn
1.8.0, hdbscan 0.8.44, scikit-fuzzy 0.5.0, ruff 0.17.3, mypy 2.4.1,
pytest 9.2.0. `mypy`: no issues in 28 source files. `pytest`: 162 passed
(23 subtests) with every extra installed and none skipped; branch
coverage 88% (2827 statements, 884 branches). `build` and
`twine check --strict` pass. The wheel installed in a fresh venv imports
from `site-packages` and its tests pass (158 tests, 12 skipped because
they need `tools/` or the extras outside the source tree). All eight
examples run. Equivalence against the dev2 reference with `--rtol 1e-9`:
16 of 20 scenarios identical; the four differing runs are `density` and
`fuzzy` under both policies, because the reference was captured without
those extras. Python 3.14 is outside the declared 3.11 to 3.13 range and
the CI matrix. `ruff` initially reported 17 findings (import order, two
unused imports, `zip()` without `strict=`, seven lines over 79 columns),
fixed in the same change.

**Parallel null stage (`SearchBudget(workers=...)`), measured 3 November
2026 on 16 CPUs, Windows 11, Python 3.14.6 (`tools/parallel_study.py`,
four blobs, seconds per default `AutoCluster().fit`).** Every run returned
the sequential recommendation, fit count and k (`same_as_first` true for
all 20 runs).

| Rows | 1 worker | 2 | 4 | 8 | 16 (`-1`) |
|---|---|---|---|---|---|
| 500 | 4.0 | 5.2 | 5.0 | 4.9 | 6.5 |
| 1000 | 11.7 | 8.5 | 6.8 | **6.0** | 6.5 |
| 2000 | 36.3 | 25.3 | 21.5 | **19.1** | 19.3 |
| 5000 | 4.4 (97 fits) | 4.1 | 3.3 | 3.1 | 6.0 |

Reading: starting processes does not pay at 500 rows (hence the 1000-row
automatic threshold); the gain is 1.9 to 2.0 times at 1000 and 2000 rows
with 8 workers, and 16 workers are no faster than 8. Amdahl's law with
the measured 81% null share predicts about 3.4 times for 8 workers, so
part of the stage is still serial (null datasets and reference distances
are built in the parent, one replicate at a time; exploration and
confirmation are sequential); the stage timings added in this version
show the split. The 5000-row line is not a speed-up: the default search
had only 97 fits because Ward and medoids failed the memory guard
(`MemoryBudgetError: workspace estimate 572.2 MiB exceeds 256.0 MiB`) and
only k-means ran, which `selection_warnings` now reports.

**Where the parallel time goes (same machine, second run, with stage
timings).** Seconds, workers 1 against 8 (first parallel fit of a process
includes starting the workers; machine variance between runs was about
30%):

| Rows | Workers | Total | Null test | Fits in the null test | Exploration |
|---|---|---|---|---|---|
| 1000 | 1 | 8.5 | 6.8 | 4.5 | 0.6 |
| 1000 | 8 (cold) | 8.0 | 6.1 | 6.0 | 0.7 |
| 2000 | 1 | 32.4 | 26.4 | 23.7 | 2.7 |
| 2000 | 8 (warm) | 18.5 | 12.2 | 12.1 | 3.1 |

The recommendation, fit count and k were again identical in all four
runs. Reading: at 2000 rows the null stage fell from 26.4 to 12.2 s
(2.2 times) and the whole fit from 32.4 to 18.5 s (1.75 times). The
parent spends about 0.1 s outside the parallel batches in the null test,
so serial preparation is not the bottleneck, contrary to the earlier
hypothesis. What remains is inside the batches: each of the 9 replicates
is a barrier of 21 uneven tasks (medoids with five restarts are the
longest), so the slowest task of every replicate sets its wall time. This
is an inference from the totals; per-task times were not measured. The
untried remedy is to submit all replicates together, longest tasks first,
and discard replicates after the first failure when counting fits. At
1000 rows the cold start of eight processes (about 2 s) cancels most of
the gain on a single fit, which is why the automatic threshold stays at
1000 and why repeated fits in one session are faster than the first.

**Other environments.** The exact comparison above is only valid in the
environment that captured the reference. A fresh capture with NumPy
2.4.4, SciPy 1.17.1 and pandas 3.0.2 differs from the reference only at
floating-point round-off (about 1e-16 on silhouettes, 1e-14 on
dispersions); statuses, recommendations, candidate identifiers and labels
are identical. `python tools/equivalence.py compare A.pkl B.pkl --rtol
1e-9` reports 20/20 identical in that case. Missing optional extras
(`hdbscan`, `scikit-fuzzy`) change the density and fuzzy scenarios.

---

# Validation

## 0.3.0.dev1 — convenience API, reporting and Gower (1 November 2026)

The original `0.3.0.dev0` source ZIP was extracted and its original sources
were executed **before modifying any implementation**. Its behavioural
reference was captured as
`docs/equivalence/reference_0.3.0.dev0_local.pkl` in this local environment.
The updated source was captured as
`docs/equivalence/reference_0.3.0.dev1.pkl` using the same versions and
single-threaded numerical environment. Both reference files have **identical
SHA-256**, `8e3a5ef53fcb917d7dd50996d37febd7e89d5427d36fb911bf2e80c1152c8207`.
`tools/equivalence.py compare` reports **20/20 IDENTICAL**, checking
recommendations, statuses, candidate tables including nested diagnostics,
labels and recorded errors, while excluding time-related columns. No
floating-point tolerance was needed. This comparison covers the default
stable Gower backend; it does **not** claim bitwise equivalence for the
new, opt-in vectorized backend.

Commands executed inside the extracted project:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python tools/equivalence.py capture docs/equivalence/reference_0.3.0.dev1.pkl
PYTHONPATH=src python tools/equivalence.py compare \
  docs/equivalence/reference_0.3.0.dev0_local.pkl \
  docs/equivalence/reference_0.3.0.dev1.pkl
```

Environment for the **regenerated** files: Python 3.13, Linux;
NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3, scikit-learn 1.8.0.
Optional density/fuzzy backends and Hypothesis were not installed. This
reference therefore records unavailable extras as unavailable; it does
not demonstrate equivalence of the optional backends in environments
where they are installed. The bundled original
`reference_0.3.0.dev0.pkl` was generated in a different environment, so
bytewise comparison with it is not an appropriate regression criterion.
The user's earlier two 1e-15-level deviations are not recharacterised
as bugs: new references remove that environment mismatch for this audit.
Pickle files should be loaded **only from trusted sources**.

### Gower microbenchmark (illustrative, not a universal performance guarantee)

One Linux process, NumPy/SciPy versions above, `OPENBLAS_NUM_THREADS=1`,
Gower on complete dense tables with 15 numeric, 12 categorical and three
asymmetric binary features, distance matrix n×n, two calls per method.
Measured wall-clock mean times (seconds):

| n | Original | Stable refactor | Vectorized opt-in | Opt-in speedup over original |
|---:|---:|---:|---:|---:|
| 100 | 0.003 | 0.005 | 0.008 | 0.40× |
| 300 | 0.025 | 0.027 | 0.016 | 1.57× |
| 700 | 0.148 | 0.157 | 0.054 | 2.75× |

The vectorized path is **slower on tiny tables**, faster for these larger
complete tables, and uses the stable path for pairwise-missing or
out-of-training-range inputs. It does not improve the model fitting step,
only Gower distance evaluation. No claim is made about model-quality gains
or peak RSS improvements. Both modes preserve asymmetric-binary and
pairwise-missing semantics within floating-point tolerance.

### Engineering checks for this development build

On the local environment above: **127 tests passed, 10 skipped**, with
16 unittest subtests passing. Skipped cases require Hypothesis or
optional density/fuzzy dependencies not available in this container.
`tools/check_source.py src/clustelia tests tools examples`: zero issues.
An offline wheel and sdist were built successfully with the installed
setuptools toolchain. A wheel was installed outside the source tree and
`cluster(..., effort="quick")` ran successfully from the installed module.
This does not replace CI testing on Python 3.11/3.12, Windows/macOS,
optional extras, or type checks using Ruff/Mypy (not installed here).

### New API contracts

Tests added in `tests/test_convenience.py` cover conservative ID detection,
effort presets and expert overrides, absence-of-recommendation reporting,
summary tables and plots, mixed Gower with independent scalar checks,
and equivalence of stable and vectorized Gower on missing values and
out-of-sample cross-distances. See `docs/CONVENIENCE_API.md` for exact
preset budgets, cautions and workflow examples. No OpenML benchmark was
rerun or recalibrated for `0.3.0.dev1`; prior substantive benchmark
results remain historical evidence, not new results.

---


## 0.3.0.dev0 (31 October 2026)

Environment: Python 3.12.3, Linux, 1 CPU, 3 GB RAM; NumPy 2.5.4, SciPy
1.18.1, pandas 3.1.0, scikit-learn 1.8.0, hdbscan 0.8.44, scikit-fuzzy
0.5.0. Engineering: `ruff` and `mypy` clean, 135 tests pass with every
extra, 93% branch coverage, `twine check --strict` passes. Equivalence:
with `n_init=1` for medoids, 20/20 runs identical to 0.2.0a3; the only
intended change is the medoid restarts (`docs/equivalence/`).

### OpenML benchmark (`tools/openml_datasets.py`, k = 2..10)

Partition drawn with seed 20261031 before any result. Labels are
compared after selection; for credit-g, cmc, credit-approval, heart-c and
adult they are outcomes, not groupings. Cells: k, ARI.

| Split | Dataset | P1 | P2 | Oracle |
|---|---|---|---|---|
| dev | balance-scale (no structure) | abstains | abstains | 0.134 |
| dev | ecoli | 5, 0.697 | 7, 0.735 | 0.495 |
| dev | vehicle | 2, 0.083 | 3, 0.073 | 0.069 |
| dev | glass | 2, 0.176 | 7, 0.201 | 0.142 |
| dev | segment | 2, 0.104 | 10, 0.472 | 0.461 |
| dev | zoo | 4, 0.875 | 6, 0.937 | 0.655 |
| dev | dermatology | 3, 0.534 | abstains | 0.686 |
| dev | credit-g | abstains | abstains | 0.007 |
| dev | cmc | abstains | abstains | 0.018 |
| dev | credit-approval | 2, 0.388 | abstains | 0.388 |
| reserved | yeast | 3, 0.012 | 3, 0.012 | 0.161 |
| reserved | wdbc | 2, 0.671 | 2, 0.671 | 0.654 |
| reserved | seeds | 2, 0.481 | 3, 0.773 | 0.773 |
| reserved | tae | abstains | abstains | 0.044 |
| reserved | heart-c | 2, 0.376 | 2, 0.376 | 0.376 |
| reserved | lymph | 2, 0.144 | abstains | 0.175 |
| reserved | mushroom | 4, 0.294 | abstains | 0.619 |
| reserved | adult | 2, 0.024 | abstains | 0.096 |

Numeric data: P2 is never more than 0.05 below P1 and is far better when
there are many classes (segment, seeds). Wide categorical tables: P2
abstains where P1 finds partitions (problem 10 in METHODS); it never made
a recommendation on the structureless control.

### Cost (`tools/cost_study.py`, default families, k = 2..8)

| Rows | Geometry | P0 | P2 | P2, B = 4 | Peak RSS |
|---|---|---|---|---|---|
| 1 000 | euclidean | 2.8 s | 9.5 s | 5.1 s | ~200 MiB |
| 1 000 | Gower | 10.2 s | 27.4 s | 14.0 s | ~210 MiB |
| 10 000 | euclidean | 3.4 s | 8.7 s | 4.2 s | ~190 MiB |
| 10 000 | Gower | 45.0 s | 48.4 s | 27.6 s | ~200 MiB |
| 50 000 | euclidean | — | 32.4 s | — | ~216 MiB |

From about 6 000 rows the default `max_pairwise_mb=256` excludes medoids
and Ward (recorded as `MemoryBudgetError` per candidate), so large
searches run k-means or k-prototypes only. Gower is the bottleneck (pandas
column loops, distances recomputed per candidate); see ROADMAP.

## Validation of 0.2.0a3

Date: 31 October 2026. Environment: Python 3.12.3 on Linux, NumPy 2.4.4,
SciPy 1.17.1, pandas 3.0.2, scikit-learn 1.8.0, threadpoolctl 3.6.0.

## Engineering checks

| Check | Result |
|---|---|
| Test suite on the sources | 117 tests: 108 passed, 9 skipped (`density`/`fuzzy` extras), 0 failed |
| Wheel installed in a fresh virtual environment, tests run outside the source tree | Same result; module resolved from site-packages |
| Equivalence with 0.1.0a3 using `SelectionPolicy(null_replicates=0)` on 4 scenarios (numeric default, families, Gower medoids, GMM BIC) | Identical statuses, recommendations, candidate tables and labels |
| `tools/check_source.py` on `src`, `tools` | 0 issues |
| Wheel and sdist build | Built; wheel includes `prototypes.py`, `structure.py`, `py.typed` |
| Examples against the sources | numeric, families, mixed, ordinal_binary, probabilistic: `recommended`; density: `incomplete` (DBCV extra absent); fuzzy: `CapabilityError` (extra absent) |

### Phase 0 re-verification (unreleased, after 0.2.0a3)

Environment: Python 3.12.3 on Linux, NumPy 2.5.4, SciPy 1.18.1, pandas
3.1.0, scikit-learn 1.8.0, threadpoolctl 3.7.0, hdbscan 0.8.44,
scikit-fuzzy 0.5.0, ruff 0.17.3, mypy 2.4.1, twine 7.0.0.

| Check | Result |
|---|---|
| `ruff check src tests tools examples` | 4 findings on 0.2.0a3, fixed; clean |
| `mypy` | 18 errors in 6 files on 0.2.0a3, fixed; clean |
| Test suite with `.[all]` | 118 passed, 0 skipped (the 9 extras tests run) |
| Wheel in a fresh venv with `[all]`, tests outside the source tree | 118 passed |
| `twine check --strict` (wheel and sdist) | Passed |
| Equivalence with unmodified 0.2.0a3 (`tools/equivalence.py`, 10 scenarios x 2 policies) | 20/20 identical |
| Equivalence with 0.1.0a3 | Not run: 0.1.0a3 sources unavailable |

`ruff format --check` would reformat 33 files; CI does not enforce it.
Windows/macOS and Python 3.11/3.13 remain covered by CI only.

## Public labelled datasets (`tools/public_benchmark.py`)

Datasets bundled with scikit-learn (no network was available for OpenML
or UCI downloads). Class labels are compared with the recommendation
after selection; they are not guaranteed to be the cluster structure of
the features. "Oracle" is k-means on standardised features told the true
number of classes. `random_state=0`, default grid k = 2..8.

| Dataset | n × p | Classes | P0 / P1: k, ARI | **P2: k, ARI** | Oracle ARI |
|---|---|---|---|---|---|
| iris | 150 × 4 | 3 | 2, 0.544 | **3, 0.592** | 0.620 |
| wine | 178 × 13 | 3 | 3, 0.897 | **3, 0.897** | 0.897 |
| breast cancer | 569 × 30 | 2 | 2, 0.671 | **2, 0.671** | 0.654 |
| digits | 1797 × 64 | 10 | 2, 0.137 | **8, 0.376** (boundary warning) | 0.468 |

With the grid widened to k = 2..15: iris, wine and breast cancer are
unchanged; on digits P0/P1 recommend k = 12 (ARI 0.556) and P2 k = 15
(Ward, ARI 0.669) with the boundary warning, because the digits gap curve
increases at every k (0.07 at k = 2 to 0.26 at k = 14). All twelve runs
recommended a partition; none of these datasets is structureless.

Reading: P2 is never worse than P1 on these four datasets and recovers
the class count on three of them (P1: two). On digits the gap statistic
does not find an elbow, a known behaviour in high dimension, and the
warning is the correct output: the number of clusters there is not
identified by this rule. Four datasets are evidence of behaviour, not a
performance estimate; no mixed-type public dataset was available offline.

Cost on public data: 246 fits for P2 against 80 for P0 with k = 2..8;
digits took 23 s for P2 against 6 s for P0 on one CPU.

## 0.2.0a2: choice of the number of clusters (policy P2)

The reserved split below was inspected scenario by scenario after the P1
run, so it no longer counts as unseen. P2 was therefore evaluated once on
a new split, `reserved2` (seeds from 200 000), after all decisions were
taken on development data.

### Development split, 6 repeats (18 structureless, 24 structured)

| Policy | False recommendations | Recommends when structured | ARI when recommended | Exact k | Mean fits |
|---|---|---|---|---|---|
| P1 (silhouette band) | 11% (2/18) | 100% | 0.909 | 75% | 193 |
| P2 before the shortlist fix | 6% (1/18) | 96% | 0.921 | 87% | 154 |
| **P2** | 6% (1/18) | 100% | 0.925 | 88% | 162 |

The single abstention of the first P2 version was a six-blob dataset
where k-means, Ward and medoids found the same partition: deduplication
kept only medoids, which failed stability. Every candidate at the
selected k is now confirmed.

### Reserved split `reserved2`, 6 repeats, run once

| Policy | False recommendations | Recommends when structured | ARI when recommended | Exact k | Mean fits |
|---|---|---|---|---|---|
| P0 (stability only) | 94% (17/18) | 100% | 0.734 | 42% | 82 |
| P1 (null test, silhouette band) | 6% (1/18) | 100% | 0.734 | 42% | 194 |
| **P2 (null test, gap statistic)** | **0% (0/18)** | 100% | **0.862** | **83%** | 153 |

Remaining P2 errors on `reserved2`: two six-blob datasets in 2-D with
spread 1.0 and 1.5 found as 3 clusters, four blobs with spread 1.5 found
as 3, and two blobs in 5-D found as 4 (ARI 0.50, where P1 had 1.0). The
gap statistic can overestimate k for blobs that are not spherical in the
standardised geometry; this is the one case where P2 lost to P1.

## 0.2.0a1 benchmark (P0 versus P1)

Scenarios, n = 300 each: isotropic blobs (k in {2, 3, 4, 6}, spread in
{0.5, 1.0, 1.5}, dimension 2 or 5), sheared blobs (k = 3), mixed-type
groups (k = 3, 15% label noise per nominal variable), and structureless
data: uniform hypercube (d = 2 or 4), one correlated Gaussian (d = 2 or 5),
independent mixed variables. Development and reserved splits use disjoint
seed ranges. Raw outcomes are in `docs/benchmark/`.

### Development split (4 repeats: 12 structureless, 16 structured)

| Policy | False recommendations | Recommends when structured | ARI when recommended | Mean fits |
|---|---|---|---|---|
| P0 (stability only) | 75% | 100% | 0.91 | 81 |
| P1, `null_scope="family"` | 0% | 100% | 0.91 | 122 |
| P1, stability 0.6 | 17% | 100% | 0.91 | 205 |
| **P1 (default: grid, stability 0.7)** | 8% | 100% | 0.91 | 193 |
| P1, stability 0.8 | 8% | 100% | 0.91 | 190 |

Decisions taken on this split: stability threshold kept at 0.7 (0.6 adds
false recommendations, 0.8 changes nothing). The family and grid scopes
cannot be separated with 12 structureless runs; the grid scope is kept
because a targeted check on 45 structureless datasets passed 5/15 uniform
datasets with the family scope versus 2/15 with the grid scope, and because
only the grid scope makes the observed and null statistics exchangeable.

### Reserved split (6 repeats: 18 structureless, 24 structured), run once

| Policy | False recommendations | Recommends when structured | ARI when recommended | Exact k | Mean fits |
|---|---|---|---|---|---|
| P0 | 78% (14/18) | 100% | 0.79 | 46% | 82 |
| **P1** | **0% (0/18)** | 100% | 0.79 | 46% | 190 |

The null-reference test removed every false recommendation on the reserved
split without rejecting any structured dataset. Recovery is identical
because the test only decides whether to recommend, not which candidate.

## Known weaknesses shown by the benchmark

- **Number of clusters.** With P1 the exact k was found in 42–46% of
  reserved structured runs because the silhouette merges close clusters;
  P2 raised it to 83% on `reserved2`. Heavily overlapping blobs are still
  merged and the gap statistic occasionally over-splits.
- **Cost.** The null test refits every explored candidate nine times on
  structured data; P2 confirms only the selected k, so it uses about 1.9
  times the fits of P0.
- **Scope.** Synthetic data (n = 300) and four bundled public datasets.
  No mixed-type public data, no large-n timing study, and no null
  reference for the probabilistic, density or fuzzy objectives.
- **High dimension.** On digits (64 features) the gap curve has no elbow
  and the selected k sits at the grid boundary.
