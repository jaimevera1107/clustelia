# Clustelia 0.3.0.dev2 — implementation and limitations

Development checkpoint, not a certified stable release. It intentionally
changes certain numerical and serialization contracts relative to dev1.

## Gap Statistic

When observed dispersion W = 0 and every reference W* > 0 is finite,
`Gap` is +infinity (the one-sided log limit), not an invalid candidate.
If any reference is missing, non-finite or nonpositive the entry is
undefined and excluded. There is no arbitrary epsilon floor. Standard error
retains the **historical ddof=0** convention. R `cluster::clusGap` uses
sample variance (`ddof=1`); a method-comparison experiment should decide
if changing this convention is justified. No claim of one universally
correct convention is made. `select_k` operates on positive infinite gaps
without requiring clipping.

## Gower comparability

- `GeometrySpec(incomparable="error")` raises on off-diagonal pairs with
  zero common comparable weight, or weighted shared-feature coverage below
  `min_comparable_share`.
- `incomparable="max"` substitutes dissimilarity 1 for such pairs. This is
  a declared **penalty policy**, not a measured distance.
- `incomparable="zero"` recreates old behavior; only for migration/testing.
- Self-distances remain 0. Coverage is a separate diagnostic returned by
  `GowerMetric.comparable_share()`. The denominator is the sum of weights
  of all *retained* contributing features, not the original feature count.
- `GowerMetric.distance_blocks` / `ReferenceGeometry.distance_blocks` yield
  `(start, stop, rectangular_distance_array)` without full n×n materialization.
  Consumers must not concatenate the whole result if memory is a constraint.
  Opt-in vectorized Gower still has different last-bit rounding.
- `null_blocks=(("a","b"),...)` permutes grouped columns together for the
  Gower reference. This is essential when columns have known joint logical
  constraints (e.g. mutually exclusive dummies). It is **not** a calibrated
  universal multivariate null and cannot guarantee false discovery rates.

## CLARA-style medoids

`CandidateSpec("clara", k=..., clara_sample_size=64, clara_samples=5)`:

1. Select a seeded sub-sample (at least `2*k+1` rows).
2. Fit alternating medoids with three restarts on the sub-sample.
3. Evaluate each candidate on all data using `ReferenceGeometry.distance_blocks`.
4. Retain the candidate with lowest full-data sum of distances.

This is a sampled heuristic; not exact PAM, and it may miss small clusters.
No default model grid is changed to include CLARA. Validation still uses
its existing metric sample and safeguards. `clara_sample_size` consumes
O(s²) memory plus O(b·k) temporary row blocks (not O(n²)).

## Reporting

`ExperimentResult.export_report("report.html" | "report.json" | ".md" |
".csv" | ".xlsx")`, `Solution.distribution_table()`,
`feature_importance_table()`, `diagnostics_table()`, `plot_heatmap()`,
`plot_projection()`, `plot_importance()`.

The differentiation scores are descriptive (`eta_squared` for numerical,
`cramers_v_squared` for categorical). They are **not significance tests**;
using these features to form clusters invalidates naïve in-sample p-values.
Projections are illustrative only; they cannot establish separation.
HTML is escaped, and JSON normalizes NaN/non-finite values to `null`.
XLSX uses optional `openpyxl` (`pip install clustelia[reports]`).

## Persistence

`result.save(path)` and `clustelia.save(result, path)` now save an
`ExperimentResult`; the older fitted `AutoCluster` saving API is retained.
`load(path, trusted=True)` returns the originally saved type. Both use
pickle, which executes arbitrary code, so never load files from untrusted
sources. Saved experiments contain original training rows, potentially
sensitive. The envelope format was bumped to 2 and package versions are
checked strictly. There is **no cross-version migration** in this release.

## Validation and unresolved work

- Dedicated edge and experience tests are in `tests/test_dev2.py`.
- Local full unittest discovery: **147 executed, 10 skipped, 0 failures/errors**; see `VALIDATION.md`.
- Candidate hashing and serialized parameters for preexisting algorithms
  deliberately exclude CLARA-specific fields; the original 20 scenarios
  are **20/20 exactly equivalent** to the dev1 local reference. New edge
  cases and explicit Gower options intentionally change semantics.
- The dev1 reference is preserved; dev2 has a new reference file in the
  same directory, captured and compared locally.
- Full cross-platform, optional dependencies, type checking, and systematic
  statistical benchmarks still require the GitHub CI matrix.
- Out-of-core data preprocessing, approximate neighbor graphs, embedding
  models and uncertainty-calibrated inference remain out of scope: no
  unsupported performance/statistical guarantee is claimed.

### Release workflow

- Create **GitHub Release** (`release.published`), not just a tag.
- Workflow: `.github/workflows/release.yml`, environment: `pypi`.
- Confirm PyPI Trusted Publisher settings match **exactly** these names.
- The 9 OS/Python jobs and lint must succeed before building; build must
  match tag version and pass `twine check --strict` before OIDC publication.
- Ruff/Mypy are separately checked by general CI; the release workflow
  does not currently enforce Mypy. Require it once it passes on all targets.
