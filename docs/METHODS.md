# Methodological decisions

This note complements the design specification. It does not claim that every
contract of the specification is implemented. Empirical results are in
[VALIDATION.md](VALIDATION.md).

## Weighted Gower

For rows i, j the distance is Σ w_c δ_ijc d_ijc / Σ w_c δ_ijc with
per-variable dissimilarities d in [0, 1]: numeric |x_i − x_j| / range
(training range, clipped to 1 for new values), nominal mismatch, ordinal
|r_i − r_j| / (L − 1) with r the position in the declared levels and L
their number, asymmetric binary mismatch. δ is 0 only for a binary joint
absence. Using the declared scale for ordinals keeps assignment stable when
a level is unobserved in training; constant variables are excluded.
Weights default to 1, are relative, and are recorded in the metadata.

Missing values (`GeometrySpec(missing="pairwise")`, Gower only): δ_ijc is
also 0 when either value is missing, which is Gower's (1971) original
treatment. Ranges and known levels come from observed values; k-prototypes
updates use observed values only (a coordinate unobserved in a cluster
stays missing and is skipped). The column-permutation null keeps the
missing pattern of each column, so the null test compares like with like
under missingness completely at random; under informative missingness the
distance can be biased and no correction is attempted. Without missing
values the complete-data code path runs unchanged.

## Gower k-prototypes

Alternating assignment (closest prototype by Gower) and coordinate-wise
update: median (numeric), mode with first-seen tie-break (nominal,
binary), lower median position (ordinal), so prototypes are always valid
values. Without asymmetric binaries the denominator is the constant total
weight, so each update minimises the cost exactly per coordinate and the
total cost is non-increasing. With asymmetric binaries the pair-dependent
denominator makes the update a heuristic. Initialisation draws prototypes
with probability proportional to the Gower distance to the closest one
already chosen; five seeded restarts keep the lowest cost; empty clusters
are reseeded with the worst-served row. Reference: Huang, Z. (1998),
*Extensions to the k-means algorithm for clustering large data sets with
categorical values*, Data Mining and Knowledge Discovery 2, 283–304.

## Alternating medoids

Alternating assignment and within-cluster medoid update on the dense
distance matrix (local descent, not PAM). Since 0.3.0.dev0 five seeded
restarts keep the lowest total distance; restart 0 draws as the former
single run. Evidence (development data only,
`tools/medoid_restarts_study.py`, 36 datasets with the true k, five base
seeds): mean ARI +0.099 (95% bootstrap CI +0.065 to +0.136), relative cost
−8.5%, ARI standard deviation across seeds −0.067; cost is never worse by
construction. On the synthetic development benchmark the P2 summary is
unchanged (0 false recommendations, ARI 0.949, exact k 100% on seeds
0–20).

## Null-reference test

Problem: on a uniform square or a single Gaussian, k-means partitions are
stable, so stability-only selection (P0) recommended a partition on 75% of
structureless development datasets.

Statistic: T = max silhouette over the explored candidates (the same
candidates, same pipelines, same metric sample size). Reference datasets:
uniform in the bounding box of the data along its principal axes
(Tibshirani, Walther and Hastie, 2001, reference (b)) for euclidean
geometry; independent permutation of each feature column for Gower. The
search passes if T on the data exceeds T on each of B replicates.

Why the maximum over the whole grid: comparing only the winning candidate
ignores that it was selected for having the highest silhouette. Measured
during development (n = 300, B = 9): refitting only the winning candidate
over its own k values passed 3 of 8 uniform squares; mirroring the k grid
of the winning family but not the choice between families passed 5 of 15
uniform datasets; mirroring the whole grid passed 3 of 45 structureless
datasets (uniform 2/15, Gaussian 0/15, mixed 1/15), consistent with the
1/(B + 1) conditional Monte Carlo resolution. Because the fitted
reference distribution is data-dependent, exchangeability and a general
10% false-recommendation error bound are not guaranteed. One test is
run per search;
retrying other candidates after a failure would multiply the chances to
pass.

Alternative family (experimental, `SelectionPolicy(null_family=
"gaussian_copula")`): a Gaussian copula fitted on the normal scores of the
data and mapped back through the empirical marginals. Every marginal and
the rank correlation are reproduced, so skewness alone is not read as
structure. Studied in November 2026 on 30 seeds per scenario (n = 160,
k-means, k 2–5, B = 19, standard scaling): on a skewed unimodal cloud
without groups the box reference recommended a partition in 22/30 runs
and the copula in 0/30; on three separated blobs 30/30 versus 27/30; on
three unbalanced blobs 28/30 versus 3/30; on three overlapping blobs
27/30 versus 20/30. The copula fixes the skewed-marginal failure at a
large cost in sensitivity on unbalanced mixtures, so `pca_uniform`
remains the default and the copula is opt-in. Both references are
data-dependent; neither carries an exchangeability guarantee.

Limits: the bound holds when the data follow the reference. Data with one
non-uniform, non-Gaussian cloud (strong skew, heavy tails) can beat the box
reference; the permutation reference treats any association between
variables as structure (two correlated numeric variables under Gower). The
test does not cover the probabilistic, density or fuzzy objectives, and
failing it does not prove absence of structure.

## Number of clusters: gap statistic

Problem: on development data the silhouette band (P1) found the exact
number of clusters in 56% of 80 structured datasets; it merges nearby
blobs because two close clusters score higher merged than apart.

Rule: W_k is the pooled within-cluster dispersion Σ_r D_r / (2 n_r) on the
metric sample with the reference distances. For every explored candidate,
Gap = mean_b log W*_b − log W, using the null fits of the structure test
(no extra fit). Each k is represented by its candidate with the largest
gap, s_k = sd(log W*) √(1 + 1/B), and the selected k is the smallest with
Gap(k) ≥ Gap(k′) − s_k′ (k′ the next explored k), as in Tibshirani,
Walther and Hastie (2001). Only candidates with that k are confirmed; the
admissibility and preference rules are unchanged.

Selection of the rule: on 80 development datasets (seeds 1000–1079; all
candidates confirmed for the comparison) the candidate rules gave mean ARI
/ exact k: silhouette band 0.817 / 0.56, maximum silhouette-minus-null
0.843 / 0.59, maximum gap 0.779 / 0.36, one-SE rule per family 0.806 /
0.75, one-SE rule over k 0.898 / 0.79; the best admissible candidate
(oracle) reached 0.922. Ad hoc rules ("largest stable k above a quality
fraction") were also tried and discarded: they reached similar ARI on some
families by over-splitting and are not published methods.

Limits: with a two-level hierarchy the one-SE rule stops at the first
level (two pairs of close blobs give k = 2), which is a defensible answer
but not the finest one. Standard scaling can make a noise-only axis as
large as an informative one; the geometry, not the rule, decides what
"close" means.

Re-evaluation in 0.3.0.dev0 (`tools/k_selection_study.py`, k-means in the
standardised space so that only the rule changes): the R `cluster::maxSE`
variants `Tibs2001SEmax` (the rule above), `firstSEmax`, `globalSEmax`
(Dudoit and Fridlyand, 2002) and `firstmax`, the same gap in the principal
subspace keeping 90% of the variance, and prediction strength (Tibshirani
and Walther, 2005; largest k with strength ≥ 0.8). On 90 development
datasets (k 2–8, dimension 2–64, 0–10 noise axes, imbalance) the paired
ARI differences against `Tibs2001SEmax` were all within noise except the
PCA gap (−0.015, CI entirely below zero); on `reserved3` (90 datasets,
run once) no rule was significantly better. Prediction strength was
slightly ahead on synthetic data but underestimated k badly on real
development data (k = 2 on digits, ARI 0.14 versus 0.47). The rule is
kept.

High dimension: the earlier "no elbow on digits" came from the grid
(k ≤ 8 with 10 classes); with k ≤ 10 the rule selects 10. The boundary
warning in `metadata["selection_warnings"]` is the intended signal, and
`metadata["selected_k_at_boundary"]` exposes it as a flag.

Boundary policy: when the gap curve is still rising at the largest
explored k the one-SE rule has not found a stopping point, and the
selected k is an unresolved upper bound. `SelectionPolicy(
boundary_k_policy="abstain")` keeps the gap table and `selected_k` as
diagnostics but spends no confirmation fit and ends the search as
`incomplete` (reason recorded in `incomplete_reasons`). This is a
governance choice, not a change to the statistic: in the November 2026
study (unbalanced blobs, no scaling) the default policy recommended
k = 5 at the boundary with ARI 0.39, and 17/30 runs reached the boundary.
The default stays `"warn"` so existing results are unchanged.

Known failure (Gower with many nominal columns): with the column-
permutation reference the gap keeps increasing with k and the null
replicates are nearly identical (s_k ≈ 0.002–0.006), so the one-SE rule
reacts to noise and the chosen k is often unstable; the search then
abstains. Seen on dermatology and credit-approval (development) and lymph
(reserved). Reverting to the silhouette would hurt synthetic mixed data
and zoo, and switching rules when the curve has no maximum is not a
published method, so nothing was changed. Candidate for the next version:
a uniform-levels reference for nominal variables, the analogue of
Tibshirani et al.'s reference (a). Workaround:
`SelectionPolicy(k_selection="silhouette")` for wide categorical tables.

Scaling: on development data standard scaling gave the best k-means ARI
with the true k on synthetic data with and without noise axes (0.940,
0.790, 0.794 for 0, 2, 10 noise axes; robust 0.925, 0.731, 0.701), and
robust scaling failed on segment (ARI 0.001, near-zero interquartile
ranges). No scaling wins on every real dataset; the default is kept.

## Density

Fitting backends: DBSCAN, OPTICS and HDBSCAN from scikit-learn. DBCV backend:
`hdbscan.validity.validity_index` 0.8.44 with `mst_raw_dist=False`.
`relative_validity_` is not used and HDBSCAN conventions are never swapped.

The DBCV dimension is that of the nonconstant euclidean reference variables,
even when the model was fitted on PCA. The same rows of the common metric
sample are evaluated, keeping noise and its weight; with sampling the score
approximates the full-data index. Assigned observations may differ between
candidates; the secondary metric `silhouette_common_support` uses the
assigned intersection of the confirmation shortlist and reports its support
and the reason when it cannot be evaluated.

Conservative rules: at least two clusters with three sampled observations
each and no duplicate reference points. Gower is not evaluated with DBCV
because no valid euclidean dimension is established for it. A missing metric
is never turned into zero, numeric and budget failures are recorded with
their cause, and a missing score does not prove absence of density
structure. No default tolerance is calibrated for this metric. No null
cluster-existence screen is performed for this objective.

## Mixtures

BIC applies only to GMM on the same input and representation. The Bayesian
mixture uses an explicit concentration prior and requires the held-out
likelihood mode, which draws one common, reproducible row split (80%
training, 20% selection by default). The value is the mean likelihood in the
coordinates of the training pipeline, not a raw-data likelihood with a
Jacobian correction. Representations are not compared with each other.

The validation fit and the full fit are different objects with stable
per-phase seeds; both costs are counted, and the retained solution holds the
labels of the full fit that was confirmed by stability. Validation is used
for selection and is not a final evaluation free of selection bias.

No null-reference test is run for mixtures, by design: mixtures model a
density, not clusters. A k = 1 versus k > 1 BIC baseline was examined and
rejected: on structureless data BIC chose k > 1 in 10/10 uniform 2-D and
10/10 log-normal 2-D datasets (a mixture approximates any non-Gaussian
density with several components), so it does not test cluster structure.

## Fuzzy C-Means

`scikit-fuzzy.cluster.cmeans` 0.5.0. Initial memberships come from a local
generator passed through `init`; NumPy's global RNG is never modified.
Assignment uses `cmeans_predict` with fixed centres and explicit
initialisation. Memberships are not calibrated probabilities.

The generalised XB_m index is

$$
XB_m = \frac{\sum_{i=1}^{n}\sum_{j=1}^{k} u_{ij}^{m}\lVert x_i-v_j\rVert^2}
{n\min_{j\ne l}\lVert v_j-v_l\rVert^2}.
$$

Lower is better; `primary_score` stores its negative so that larger is always
better internally. The fuzzifier m is fixed within an objective (for m = 2 it
matches the usual quadratic form). The FCM objective, FPC, partition entropy
and iterations are published as separate diagnostics and never averaged.
Coincident centres do not produce an artificial finite score.

## Tie-break cost

The specification orders practically equivalent candidates by stability,
coverage, cost and identifier. Since 0.1.0a3 cost is the number of fits used
by the candidate (exploration, validation and confirmation), the same unit
as `SearchBudget.max_fits`. Wall-clock time made the recommendation depend
on machine load; it remains available as `cost_tiebreak="fit_seconds"`.

## Primary sources

- [DBCV code and documentation in hdbscan](https://hdbscan.readthedocs.io/en/latest/_modules/hdbscan/validity.html).
- [scikit-learn 1.8 HDBSCAN](https://scikit-learn.org/1.8/modules/generated/sklearn.cluster.HDBSCAN.html).
- [scikit-learn 1.8 BayesianGaussianMixture](https://scikit-learn.org/1.8/modules/generated/sklearn.mixture.BayesianGaussianMixture.html).
- [scikit-fuzzy Fuzzy C-Means API](https://scikit-fuzzy.readthedocs.io/en/latest/api/skfuzzy.cluster.html).
- Gower, J. C. (1971), *A general coefficient of similarity and some of its
  properties*, Biometrics, 27(4), 857–871.
- Tibshirani, R., Walther, G. and Hastie, T. (2001), *Estimating the number
  of clusters in a data set via the gap statistic*, JRSS B, 63(2), 411–423.
- Ben-David, S., von Luxburg, U. and Pál, D. (2006), *A sober look at
  clustering stability*, COLT.
- Pal, N. R. and Bezdek, J. C. (1995), *On cluster validity for the fuzzy
  c-means model*, IEEE Transactions on Fuzzy Systems, 3(3), 370–379.
