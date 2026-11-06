# Implemented scope and next work

State of **0.1.0** (first public release; internal snapshots 0.2.0a1–0.3.0a1 preceded it).

| Milestone | Implemented | Pending |
|---|---|---|
| M1: contracts and core | Numeric/nominal schema, explicit geometry, preserved pipelines, KMeans, Ward, medoids, GMM, density with optional DBCV | Public precomputed distances, custom policies, full report |
| M2: families | MiniBatchKMeans, BIRCH, BisectingKMeans, Spectral, OPTICS, HDBSCAN, optional FCM, Bayesian mixture; Ward/OPTICS structures | Public hierarchies for more adapters, sensitivity |
| M3: categorical/mixed | Nominal, ordinal and asymmetric binary semantics; weighted Gower; medoids and Gower k-prototypes (k-modes on nominal data); Gower pairwise deletion of missing values | Gap reference for nominal data |
| M4: differentiation | Deduplicated alternatives, manual selection, `ambiguous` status, common support, **null-reference test**, **gap-statistic choice of k** | Constraints, consensus, null references for mixtures and density |
| M5: validation/distribution | Installable typed package, 173 tests under unittest (about 177 items under pytest; 88% branch coverage measured on 3 November 2026, Windows 11 / Python 3.14, was 93% at 0.3.0.dev0), property tests, examples, persistence, CI matrix, build check, release workflow, synthetic benchmarks with three reserved splits, 18-dataset OpenML benchmark with a reserved split | Cost reductions for large n, publication (MIT license set) |

## Next milestone

1. **Gap reference for nominal variables** (problem 10). With many
   nominal columns the column-permutation reference makes the gap grow
   with k and the search abstains (dermatology, credit-approval, lymph,
   mushroom). Evaluate a uniform-levels reference (analogue of Tibshirani
   et al.'s reference (a)) on development data, then on a fresh reserved
   OpenML split (the current one is retired).
2. **Cost (priority).** Profile (default `AutoCluster().fit` on 4 blobs,
   1 CPU, 3 November 2026): n = 500 takes 5.5 s, n = 1000 13.1 s and
   n = 2000 42 s, always 246 fits (21 exploration, 36 confirmation, 189
   null-reference). At n = 1000 the null test is 10.6 s of 13.1 s (81%);
   by cost, alternating medoids is 4.4 s (3.3 s in the `_update` gather
   loop), the silhouette on the metric sample 3.4 s (213 calls), k-means
   1.5 s and Ward 1.2 s. `cluster(effort="quick")` stops at 56 fits and
   ends `incomplete` on the same data. Options: (a)
   candidate-level parallelism of the null stage: implemented in
   `SearchBudget(workers=...)` (results independent of the worker count;
   measured: 1.75 to 2.0 times with 8 workers at 2000 rows on 16 CPUs,
   see VALIDATION. The parent-side preparation is only about 0.1 s, so the
   loss is the per-replicate barrier of uneven tasks: next, submit all
   replicates together, longest tasks first, keeping the fit counts of the
   sequential path; then parallelise exploration and confirmation (about
   18% of the sequential time)), (b) cap
   the rows used by the null/gap stage for large n (valid test, flat
   cost; needs a development benchmark), (c) drop or demote medoids in
   the euclidean default grid (largest per-fit cost; changes results),
   (d) replace the medoid gather by a matrix product (about 25% of the
   run; floating-point sums change, so ties may change). Original plan:
   profile first; then, keeping results identical:
   one distance matrix shared by candidates and restarts, NumPy-vectorised
   Gower, candidate-level parallelism. Then `effort` presets (fast /
   standard / thorough) with measured time-quality trade-offs, a visible
   default time budget and a warning when the memory guard drops
   families.
3. **Untested branches** (coverage 93% with branches): null-reference
   budget exhaustion mid-replicate (only reachable through the soft
   deadline), backend failure inside a null replicate, k-prototypes
   empty-cluster repair, `null_scope="family"` rejection bookkeeping.
4. **Release:** TestPyPI then PyPI (license MIT is set).

## Decisions kept

- Alternating medoids is local descent, never presented as PAM; five
  seeded restarts by default (0.3.0.dev2).
- Gower k-prototypes is not Huang's k-prototypes and says so.
- No universal score ranks different objectives or geometries.
- A recommendation must beat structureless reference data (compact
  objective); one test per search, abstention on failure.
- The number of clusters comes from the gap statistic (one-SE rule), not
  from the silhouette; the silhouette only ranks candidates sharing k.
- A reserved split is evaluated once per methodology version and is
  retired once inspected.
- DBCV without a compatible backend or geometry yields `incomplete`.
- Density and fuzzy keep several admissible alternatives by default.
- Budgets count every pipeline fit, not only exploration.
- The deadline is soft and memory guards are preventive.
- Dependent data need group/block protocols that are not implemented.
- Pickle keeps data and requires explicit trust; no automatic migration.
- Gower missing values: pairwise deletion on explicit request only;
  the default still rejects them.
- No k = 1 BIC baseline for mixtures: it flags non-Gaussian structureless
  densities as clustered.
- Rules for k must be published; switching rules on curve shape was
  rejected as ad hoc.
