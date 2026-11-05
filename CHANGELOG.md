# Changelog

All notable changes are documented here. The project follows
[Semantic Versioning](https://semver.org/); 0.x releases may change the API.

## Unreleased — null family and boundary policy (follow-ups of the
## November 2026 validation study)

- `SelectionPolicy(null_family=...)`: choose the structureless reference
  explicitly. `"auto"` (default) keeps the geometry default
  (`pca_uniform` for Euclidean, `permutation` for Gower).
  `"gaussian_copula"` is a new EXPERIMENTAL family that preserves every
  marginal and the rank correlation; numeric features only, rejected at
  plan time for Gower geometry. See `docs/METHODS.md`.
- `SelectionPolicy(boundary_k_policy="abstain")`: when the gap statistic
  selects the largest explored `k`, withhold confirmation and end the
  search as `incomplete` instead of recommending an unresolved upper
  bound. Default `"warn"` is unchanged.
- Result metadata and `summary()` now carry `null_family`, `selected_k`,
  `selected_k_at_boundary` and `boundary_k_policy` so abstention causes
  and the family used are visible without reading `structure_test`.
- Internal: null generators are registered per family in
  `structure._GENERATORS`; adding a family no longer edits
  `generate_null`.
- `tools/null_reference_study.py`: paired 30-seed comparison of the
  reference families (recommendation rate with Wilson intervals, exact
  k, conditional ARI, boundary abstentions). Results in
  `docs/benchmark/null_reference_study_30.json` and `docs/VALIDATION.md`.
- Behavioural equivalence against `main` with default settings: 20/20
  identical (`tools/equivalence.py`, exact mode).

## Unreleased — audit corrections (review before releasing)

- Align quick budget with the complete default numeric candidate grid.
- Mark null-backend failures and unavailable Gap statistics as incomplete.
- Rank every objective using its oriented primary score within experiment.
- Use bias-corrected squared Cramér V for categorical differentiation.
- Account for all attempted parallel fits when a batch fails.
- Expose validation scope and clarify conditional null-reference evidence.
- Make static typing a required CI gate and install Hypothesis in test CI.
- Add targeted regression tests and document retained statistical limits.

## 0.3.0a1 — 3 November 2026

- Repository-preparation cleanup: preserve the source algorithms; keep one
  frozen current equivalence snapshot, archive redundant historical
  snapshots separately, and exclude benchmarks/pickles/tests/tools from
  package distributions. No change to the clustering numerical engine.


First publishable alpha. It consolidates the 0.3.0.dev0 to dev2 changes
listed below it; the import package is `clustelia` (formerly `clustera`).


- Handle exact zero observed dispersion as an infinite, legitimate Gap
  when positive finite references exist; reject undefined reference cases.
- Reject Gower pairs without enough comparable information by default,
  with explicit max-penalty and legacy-zero policies, comparability
  diagnostics, and chunked row-distance generator.
- Preserve within-block dependencies when generating Gower null permutations.
- Add independent null replicate override to the convenience `cluster()` API.
- Support safe-contract (trusted-only) saving/loading of `ExperimentResult`.
- Add descriptive categorical distribution and feature differentiation,
  balance diagnostics, PCA and heatmap visualizations, and HTML, JSON,
  Markdown, CSV and XLSX exports.
- Add opt-in CLARA sampled medoids with blockwise full-dataset evaluation;
  no longer require an n-by-n distance matrix for this estimator.
- Gate PyPI publishing on GitHub Release, CI matrix, source lint, version
  match, package construction and metadata verification.
- Set BSD-3-Clause license and project URLs. Historical candidate IDs
  remain stable; new equivalence baseline verified 20/20 exact scenarios.
- `tools/equivalence.py compare --rtol TOL` compares captures made in
  different NumPy/SciPy/pandas environments: floats within tolerance,
  statuses, recommendations, identifiers and labels still exact.
  Without `--rtol` the comparison stays exact (same environment only).
- `tools/verify_all.py` runs the CI checks (ruff, mypy, tests with branch
  coverage, extras, build, twine, clean-venv install, equivalence,
  examples) on a local machine and writes `verification_report.md`; see
  `docs/LOCAL_VERIFICATION.md`.
- **Parallel null-reference stage.** `SearchBudget(workers=...)` runs the
  null-reference refits (about 80% of the default cost) in worker
  processes: `None` chooses automatically (sequential below 1000 rows, else
  up to 8 workers), `1` is sequential, `-1` uses every CPU. Seeds, fit
  counts, candidate identifiers, tables and labels do not depend on the
  value (tested with 1, 2 and 3 workers, numeric and Gower); a batch that
  the fit budget cannot cover runs sequentially as before. New
  `FitExecutor.fit_apply`, `metadata["parallel_workers"]` and
  `tools/parallel_study.py` to measure the effect on a machine.
- `metadata["selection_warnings"]` (and the log) now report an algorithm
  whose candidates all failed, e.g. dense-matrix families above the memory
  guard: at 5000 rows the default search used to drop Ward and medoids
  silently (`MemoryBudgetError`) and recommend from k-means alone.
- `metadata["stage_seconds"]` and `metadata["structure_test_seconds"]`
  give wall-clock time per fit stage; `tools/parallel_study.py` prints
  them.
- Declare `joblib` (already required by scikit-learn) as a direct
  dependency, since the worker processes import it, and tell `mypy` it
  ships no type information.
- Make `test_timeout_no_false_recommendation` independent of the operating
  system timer resolution (it failed intermittently on Windows CI), and
  tell `mypy` that matplotlib and openpyxl are optional.
- Fix two `mypy` errors found by it on Windows / Python 3.14
  (`results.py` reused a variable with two types; `structure.py` lacked an
  annotation). No behaviour change.
- Fix the `ruff` findings reported on Windows / Python 3.14: import order
  in six modules, two unused imports in `tests/test_dev2.py`, seven long
  lines in `tests/test_convenience.py`, `zip(strict=True)` in the
  equivalence tool.

## 0.3.0.dev1 — unreleased (1 November 2026)

### Added
- `cluster()` convenience entry point with `quick`, `balanced` and
  `thorough` effort presets, without changing the expert defaults.
- Conservative, reviewable identifier detection for complete unique
  ID-named DataFrame columns; explicit schemas take precedence.
- `summary()`, `report()`, `ranking_table()`, `sizes_table()`,
  `profile_table()`, `plot_sizes()`, `plot_profile()` and
  `plot_candidates()` for accessible interpretation.
- Cluster-level profiles now compare medians, categorical lift and binary
  shares to the full dataset. Matplotlib remains an optional `viz` extra.
- An optional `gower_backend="vectorized"` for complete-case dense data,
  using compiled SciPy distance kernels; stable Gower remains the default.
- 20/20 behaviourally identical before/after local equivalence snapshots,
  with identical SHA-256. See `docs/VALIDATION.md`.

### Limitations
- The vectorized Gower backend is opt-in because last-bit floating-point
  differences may change the handling of ties. Missing and extrapolated
  values fall back to the stable backend.
- Selection quality has not been re-benchmarked on external data.

## 0.3.0.dev0 — previous development snapshot

### Renamed
- The distribution and import package are now **`clustelia`** (formerly
`clustera`). The main class stays `AutoCluster`. The base exception is
`ClusteliaError`. Pickles saved under the old name are not loadable; re-fit
and save again. Entries below this line keep the historical name.

Roadmap phases 0 to 5. Methodology decisions used development data only;
each reserved split was run once (see `docs/VALIDATION.md`).

### Breaking
- **`medoids` runs five seeded restarts by default** (`n_init=5`, was a
  single run) and keeps the lowest total distance. On 36 development
  datasets: ARI against the truth +0.099 (95% bootstrap CI +0.065 to
  +0.136), cost -8.5%, seed dependence -0.067 ARI sd. Results of any
  search with medoids differ from 0.2.0a3; with `n_init=1` they are
  identical (checked on the 20 equivalence runs). Restart 0 draws exactly
  as the former single run.
- Files saved with 0.2.0a3 do not load (the persistence envelope pins
  the package version; no migration, as documented).

### Added
- **Gower pairwise deletion of missing values:**
  `GeometrySpec(kind="gower", scaling="none", missing="pairwise")`
  compares each pair on the variables observed in both (Gower, 1971).
  Propagated to k-prototypes (observed-value medians/modes), `predict`
  and `describe()` (new per-cluster `"missing"` fractions). Default stays
  `"error"`; without missing values the complete-data code path runs
  unchanged. Rows with no observed feature are rejected.
- `AlternateMedoids(n_init=...)` and `clustera.medoids.DEFAULT_RESTARTS`.
- `tools/openml_datasets.py`: frozen registry of 18 OpenML datasets (id,
  version, md5 checked on load), development/reserved partition drawn by
  seed 20261031 before any result, fixed preprocessing protocol.
- `tools/public_benchmark.py --source openml --split ...`: mixed tables,
  NMI, Gower k-prototypes oracle for non-numeric data.
- `tools/k_selection_study.py`: published k rules (R `cluster::maxSE`
  variants, gap in a PCA subspace, prediction strength) with development
  and `reserved3` splits.
- `tools/medoid_restarts_study.py`, `tools/cost_study.py`,
  `tools/equivalence.py`, equivalence references for 0.2.0a3 and
  0.3.0.dev0.
- Property-based tests of Gower (hypothesis), missing-value tests and
  medoid restart tests; `hypothesis` and `pytest-cov` in the `dev` extra.

### Fixed
- `ruff check` and `mypy` clean (4 lint findings and 18 type errors on
  0.2.0a3; no behaviour change).

### Kept after evaluation (no change)
- Gap statistic with the Tibshirani 2001 one-SE rule (P2): no published
  alternative was significantly better on development or `reserved3`.
- Standard scaling as euclidean default; no k=1 BIC baseline for mixtures.

## 0.2.0a3 — 2026-10-31

### Added
- `metadata["selection_warnings"]`: warns (and logs) when the gap
  statistic selects the largest explored k, since the true number of
  clusters may be larger.
- `tools/public_benchmark.py`: benchmark on public labelled datasets
  (iris, wine, breast cancer, digits; bundled with scikit-learn), with a
  k-means oracle told the true number of classes. Results in
  `docs/VALIDATION.md`.

## 0.2.0a2 — 2026-10-31

Changes how the compact objective chooses the number of clusters.

### Changed
- **Gap-statistic choice of k (policy `P2-alpha1`).** After the null test
  passes, `k` is chosen with the gap statistic and its one-standard-error
  rule, reusing the null fits, and only candidates with that `k` are
  confirmed. Silhouette-based selection merged nearby blobs. Controlled by
  `SelectionPolicy.k_selection` (`"auto"` by default; `"silhouette"` gives
  the 0.2.0a1 behaviour).
- When the null test fails, no confirmation fit is spent.
- The null test result is also stored in `metadata["structure_test"]`;
  new `metadata["gap_table"]`, `metadata["selected_k"]` and
  `metadata["k_selection"]`.

### Added
- `within_dispersion` column (pooled within-cluster dispersion `W` on the
  metric sample).

## 0.2.0a1 — 2026-10-31

Completes the mixed-data milestone (M3) and adds a null-reference test to
the selection policy. With `SelectionPolicy(null_replicates=0)` the
results are identical to 0.1.0a3 on the equivalence scenarios (statuses,
recommendations, candidate tables and labels).

### Added
- **Ordinal and asymmetric binary variables.** `Schema(ordinal={column:
  levels}, binary=[...])`, validated at fit and assignment time.
- **Weighted Gower.** `GeometrySpec(kind="gower", weights={...})`; ordinal
  contributions use declared level positions and asymmetric binaries skip
  joint absences (Gower, 1971).
- **`kprototypes` adapter**: Gower k-prototypes (median / mode /
  lower-median level), assignable, O(n·k) memory, five seeded restarts.
  Exactly k-modes on nominal-only data.
- **Null-reference test (policy `P1-alpha1`).** Every explored candidate is
  refitted on `null_replicates` structureless datasets (PCA-box uniform for
  euclidean, column permutation for Gower) and the best silhouette on the
  data must beat every replicate. New `structure_test` column,
  `structure` fit stage, `SelectionPolicy.null_replicates` and
  `null_scope`, and `reserved_structure_fits` in `plan()`.
- `AutoCluster.predict(X)` assigns new rows with the recommended pipeline.
- `describe()` profiles ordinal and binary variables.
- `tools/benchmark.py`: reproducible synthetic benchmark with development
  and reserved splits; results in `docs/VALIDATION.md`.

### Changed
- Default policy is `P1-alpha1`; `SelectionPolicy.p0()` now returns the
  stability-only P0 policy (`null_replicates=0`, version `P0-alpha3`).
- Default `SearchBudget.max_fits` is 320 (was 160) to hold the null
  reference reserve.
- Default Gower families are `medoids` and `kprototypes` (was medoids
  only), so mixed-data results differ from 0.1.0a3.
- Any ordinal or binary feature selects Gower by default; euclidean
  geometry rejects them.

## 0.1.0a3 — 2026-10-31

Hardening release. Fitted results are identical to 0.1.0a2 on the
equivalence scenarios (statuses, recommendations, candidate tables and
labels); the changes below are the only intended differences.

### Changed
- **Deterministic tie-break.** Cost now counts fits
  (`SelectionPolicy.cost_tiebreak="fit_count"`) instead of wall-clock
  seconds, so the recommendation is reproducible across runs. The previous
  rule is available with `cost_tiebreak="fit_seconds"`. The methodology
  label is now `P0-alpha3`.
- `NoRecommendationError` also derives from `AttributeError`, so
  `hasattr(model, "labels_")` returns `False` instead of raising.
- Pairwise memory guards raise `MemoryBudgetError` (a `MemoryError`
  subclass); every deliberate error derives from `ClusteraError`.
- Fitted attributes are set only after a successful search; a failed `fit`
  no longer leaves `n_features_in_` behind.
- Solutions share one read-only training table and medoid pipelines store
  only their prototypes (a 20-candidate model went from 19.7 MB to 11.0 MB
  when pickled).
- `save` writes atomically and rejects non-`AutoCluster` objects; `load`
  rejects files that are not clustera envelopes and names the mismatched
  versions.
- Replica rejection reasons accumulate instead of keeping only the last.
- Shortlisted candidates whose confirmation was cut by the budget record
  `"confirmation incomplete within budget"`.
- Documentation, examples and messages are in English.

### Fixed
- NumPy integers (e.g. `k_range=np.arange(2, 9)`, `random_state=np.int64(7)`)
  were rejected; integer parameters now accept any integral type and are
  normalised to `int`. Booleans are still rejected.
- An empty or invalid `k_range` produced a misleading error.
- `Schema(numeric="age")` was silently split into characters; bare strings
  are now rejected.
- `external_evaluate` validated `y` inside its loop.

### Added
- Public `ClusteraError`, `MemoryBudgetError`, `ExperimentResult`,
  `Solution`, `Recommendation` and `Membership` exports; `py.typed`.
- `fits_used` column per candidate; `ExperimentResult.candidate_ids` and
  `fitted_algorithms`.
- Single-sourced version, `dev` extra, ruff/mypy/pytest configuration, CI on
  Linux/Windows/macOS with lint, type check and package build, a tag-driven
  Trusted Publishing workflow, `CONTRIBUTING.md` and
  `tools/check_source.py`.
- 18 regression tests (71 in total).

### Internal
- `AutoCluster.fit` was split into planner, fit executor, objective scorers
  (strategy per objective), stability confirmer and selection functions.
  Estimator construction uses a factory registry; the reference geometry
  delegates to euclidean/Gower metric strategies.

## 0.1.0a2 — 2026-10-31

- Eight additional families: MiniBatchKMeans, BIRCH, BisectingKMeans,
  Spectral, OPTICS, HDBSCAN (scikit-learn), FCM (scikit-fuzzy) and
  BayesianGaussianMixture.
- Configurable automatic search by family; expert input uses the same engine.
- Optional, versioned DBCV with domain validation and explicit support rules.
- Gaussian/Bayesian mixtures with reserved validation likelihood,
  transformations fitted on training rows only and both costs recorded.
- Separate fuzzy objective: XB_m with fixed fuzzifier, descriptive FPC and
  entropy.
- `ambiguous` status for density/fuzzy without an explicit tolerance.
- Distinct membership semantics; Ward structure and OPTICS reachability.
- Additional confirmation controls on coverage and assignment/noise
  agreement.
- Secondary diagnostic on common support, fit breakdown and failure warnings.
- Centralised version and explicit rejection of incompatible pickles.

## 0.1.0a1 — 2026-10-30

First experimental release derived from the notebook and the methodological
specification: `src` layout, immutable contracts and the `AutoCluster` API;
reference geometry independent of the fitting space; KMeans, Ward,
alternating medoids, GaussianMixture and exploratory DBSCAN; Gower for mixed
tables; confirmation by subsampling and seeds, P0 filters and Pareto
frontier; preserved pipelines, manual selection without refitting and
profiles; fit budgets, soft deadline and pairwise memory guards; per-candidate
failures and warnings; post-hoc external evaluation; trusted persistence.
