# Convenience API and interpretable outputs — 0.3.0a1

The **expert API** (`AutoCluster`, explicit `Schema`, `GeometrySpec`,
`SelectionPolicy` and `SearchBudget`) is still supported with unchanged defaults.
This release adds an independent convenience entry point that **does not**
force a partition when the validation policy abstains.

## End-to-end usage

```python
import pandas as pd
from clustelia import cluster

frame = pd.read_csv("observations.csv")
result = cluster(frame, effort="balanced", random_state=42)
print(result.summary())                 # dictionary with selection status
print(result.report())                  # Markdown, not a classifier guarantee
print(result.ranking_table(top=12))     # candidate-level comparison

solution = result.recommended_solution()
if solution is not None:
    print(solution.sizes_table())       # cluster, size, share, noise
    print(solution.profile_table())     # tidy feature × cluster profiles
    ax = solution.plot_profile("income")   # optional matplotlib
    solution.plot_sizes()               # optional matplotlib
    result.plot_candidates()            # compact objective only
else:
    print("No sufficiently supported partition was recommended.")
```

For a shorter run use `effort="quick"` and for a more extensive check use
`effort="thorough"`. **These are computational budgets and evidence settings,
not calibrated confidence levels.** Their parameters:

| Effort | Null replicates | Subsamples | Extra seed repeats | Shortlist | Max fits | Candidates | Timeout (soft) | Metric sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| quick | 2 | 3 | 1 | 3 | 100 | 21 | 120 s | 500 |
| balanced | 9 | 10 | 3 | 5 | 320 | 40 | 600 s | 1 000 |
| thorough | 19 | 15 | 5 | 7 | 800 | 75 | 1 800 s | 1 500 |

`quick` covers the entire default compact grid (21 numerical
candidates) before attempting validation. Custom candidate grids can still be
truncated, and the status will then be `incomplete`. `quick` uses less null
evidence than `balanced` and can recommend structures
that fail a more extensive validation. A bigger budget does not make the data
clusterable. Explicit `selection=` and `budget=` override the preset; other
`AutoCluster` arguments (`geometry`, `objective`, `families`, `k_range`, etc.)
are forwarded unchanged.

## Identifier detection: deliberately conservative

`cluster` defaults to `identifiers="auto"` for **DataFrames only**. A column is
automatically excluded as an ID only if (a) its name contains an explicit
ID-like token (e.g. `customer_id`, `record_id`, `UUID`), (b) values are
complete, and (c) values are unique. A numeric measurement or categorical
column with unique values **is not** automatically excluded merely because
it is unique. Duplicated or missing ID-like columns remain features unless
explicitly declared; they should be reviewed by the analyst.

`cluster(frame, identifiers=None)` disables detection;
`cluster(frame, identifiers=["code", "person"])` declares identifiers
explicitly. For explicit numeric, ordinal, asymmetric binary and nominal
semantics, pass `schema=Schema(...)` and leave `identifiers="auto"` at its
default: explicit schema always wins. All declared identifiers are validated
under the core requirement of complete unique combinations. Positional NumPy
arrays have no automatic ID detection. `detect_identifiers(frame)` can be used
without fitting to preview the names the heuristic would exclude.

Auto-detection is intentionally limited: no unique-value percentages,
monotonicity tests or inferred personal identifiers are used. The heuristic
is not a semantic guarantee and should be reviewed for real data.

## Interpreting the results

`result.summary()` includes the selection status, candidate/fits counts,
recommended family and k if available, warnings and excluded IDs. It does
not silently promote a fitted candidate to a validated recommendation.
`result.report(top=5)` returns Markdown suitable for notebooks or documents.
`result.ranking_table()` displays candidate rows with the recommendation
flag and key comparable diagnostics. **Scores across different objectives
are not directly comparable**, and the presence of a high silhouette alone
does not override an abstaining selection policy.

`solution.profile_table()` contains cluster × feature summaries using input
units: median and quartiles for numeric columns, most common category and its
share for categorical/ordinal columns, proportion of ones for asymmetric
binary columns, missing share, noise indicator, cluster size, and contrasts against the
whole input (numeric median difference, category lift or binary-share
difference). Its `describe()`
method remains available unchanged and retains complete categorical shares.
`plot_profile(feature)` and `plot_sizes()` are observational diagnostics,
not evidence that the partition is objectively true. Install the optional
`viz` extra for plotting: `pip install 'clustelia[viz]'`.

## Gower: stable versus vectorized

The default `GeometrySpec(kind="gower", scaling="none")` uses **stable**
feature-order summation, matching the historical reference computations.
For complete cases, the optional backend uses SciPy compiled weighted
Manhattan/Hamming operations plus binary dot products:

```python
from clustelia import cluster, GeometrySpec
result = cluster(
    frame,
    geometry=GeometrySpec(kind="gower", scaling="none",
                          gower_backend="vectorized"),
    effort="balanced",
)
```

The vectorized backend **falls back to stable Gower** on pairwise-missing
samples and values outside fitted numeric ranges (to keep clipping correct).
The batched/compiled backend avoids creating an n×m×p array, but still creates
an n×m dense output and is subject to the same pairwise-memory guard.
The summation order differs by floating-point rounding and can affect
exact-tie medoid paths; it is deliberately opt-in. The regression suite
checks numerical agreement to tolerance, but **does not assert bitwise
identity for that opt-in mode**.

No changes to the `AutoCluster` expert defaults or existing selection policy
are required for using the new reporting interface.
