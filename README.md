<div align="center">

# Clustelia


<p align="center">
  <img
    src="https://raw.githubusercontent.com/jaimevera1107/clustelia/main/docs/Clustelia%20Hexagon%20Cluster%20Logo.png"
    alt="Clustelia logo"
    width="360"
  />
</p>


### Clustering with evidence, not guesswork.

**Explore, compare, and validate clustering solutions — without forcing a partition.**

Clustelia is a Python library for automatic clustering of numerical, categorical, ordinal, binary, and mixed data. It searches multiple algorithms, evaluates candidates in a common reference geometry, and can **abstain from recommending a solution** when the evidence is insufficient.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-526477?style=flat-square)](LICENSE)
![Development](https://img.shields.io/badge/Development-0.3.0a1%20%7C%20alpha-7B8794?style=flat-square)
[![PyPI](https://img.shields.io/badge/PyPI-clustelia-2A6EA6?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/clustelia/)

[**Quick start**](#quick-start) · [**Methodology**](#how-selection-works) · [**Mixed data**](#mixed-data) · [**Documentation**](#documentation) · [**Contributing**](#development)

</div>

---

> [!NOTE]
> **Development status — `0.3.0a1`.** This README describes the alpha source version, which may be newer than the package currently published on PyPI. Install from the development checkout to use the APIs shown below until a matching release is published. Clustelia is under active validation and is **not yet a production guarantee**.

## Why Clustelia?

Most clustering workflows can produce labels even when the data contain no meaningful cluster structure. Clustelia treats choosing **not** to recommend a partition as a valid analytical result.

| Capability | What it means |
|:--|:--|
| **Evidence-based selection** | Compare observed structure against structureless reference data before recommending a compact clustering solution. |
| **Multiple algorithms** | Explore compact, probabilistic, density-based, and fuzzy clustering families under explicit objectives. |
| **Mixed-data support** | Model numeric, nominal, ordered, and asymmetric binary features with Gower geometry. |
| **Reproducible experiments** | Control random seeds, search budgets, candidate specifications, and confirmation runs. |
| **Interpretable results** | Inspect ranked candidates, selection reasons, cluster profiles, and fitted pipelines — including when the search abstains. |

Clustelia does not treat a good score as proof that clusters exist. **A fitted candidate is not automatically a validated recommendation.**

## Installation

**Requirements:** Python 3.11 or newer.

For the version currently available on PyPI:

```bash
python -m pip install clustelia
```

For the **`0.3.0a1` source version** described in this README, use a checkout containing that code:

```bash
# Run inside the project directory
python -m pip install -e ".[dev,all]"
```

Install only the extras you need:

| Extra | Adds |
|:--|:--|
| `clustelia[viz]` | Visualizations with Matplotlib |
| `clustelia[reports]` | Excel-related functionality with openpyxl |
| `clustelia[density]` | DBCV validity index through the optional HDBSCAN package |
| `clustelia[fuzzy]` | Fuzzy C-Means with scikit-fuzzy |
| `clustelia[all]` | All optional features |
| `.[dev]` from source | Testing, linting, type-checking, and package build tools |

> [!IMPORTANT]
> The **HDBSCAN clustering model** uses scikit-learn; the `density` extra is for DBCV. Optional backends with exact version requirements are checked for compatibility rather than silently substituted.

## Quick start

The `cluster()` entry point provides sensible effort presets while preserving access to the full clustering engine.

```python
from sklearn.datasets import load_iris
from clustelia import cluster

X = load_iris(as_frame=True).frame.drop(columns="target")

result = cluster(X, effort="balanced", random_state=42)

print(result.summary())
print(result.ranking_table())

solution = result.recommended_solution()
if solution is None:
    print("No sufficiently supported clustering solution was recommended.")
else:
    print(solution.profile_table())
    # solution.plot_sizes()  # requires the 'viz' extra
```

Three presets balance runtime against the amount of validation evidence:

| Preset | Null replicates | Subsamples | Max. fits | Intended use |
|:--|--:|--:|--:|:--|
| `quick` | 2 | 3 | 100 | Fast exploration; weaker validation evidence |
| `balanced` | 9 | 10 | 320 | Default exploration and confirmation |
| `thorough` | 19 | 15 | 800 | More extensive evidence; higher computational cost |

*Effort is a computational preset, not a calibrated confidence level.* You can override its selection policy or budget explicitly. See [Convenience API](docs/CONVENIENCE_API.md) for parameters, automatic identifier detection, summaries, plots, and exports.

### Understanding the result

Clustelia can return these selection states:

| Status | Interpretation |
|:--|:--|
| `recommended` | A candidate passed the applicable selection and confirmation criteria. |
| `ambiguous` | The available evidence does not support a unique recommendation. |
| `no_admissible_solution` | No candidate passed the applicable admissibility criteria. |
| `incomplete` | The search/confirmation was truncated, a required statistic was unavailable, or a reference-validation fit failed. |

Always check for a recommendation before accessing recommended-model convenience properties. With `AutoCluster`, `model.labels_`, `model.best_estimator_`, `predict()`, and `fit_predict()` raise `NoRecommendationError` when none exists. You can still inspect individual fitted candidates through `result.select(candidate_id)`; manual selection **does not validate** them.

## How selection works

For the **compact** objective, the default P2 policy makes three distinct decisions:

1. **Is there evidence of structure?** Candidate partitions are compared with structureless reference datasets (uniform references in a principal-axis box for Euclidean geometry, or column permutations for Gower). The observed best silhouette must exceed its reference counterparts.
2. **How many clusters?** The gap statistic uses the null-reference fits to choose `k`, rather than relying only on the silhouette, which can favor a coarser partition.
3. **Which fitted candidate is most reliable?** Eligible candidates are confirmed through subsampling and alternative random seeds. The final choice considers stability, coverage, performance, and computation cost.

The reference comparison is a statistical screening rule, **not proof of the existence or absence of natural clusters**. With nine null replicates, the nominal Monte Carlo resolution is `1 / (9 + 1) = 10%` under exchangeability. The fitted data-dependent references do **not** guarantee exact calibration or a universal 10% false-recommendation rate. Increasing the evidence budget does not make every dataset clusterable.

<details>
<summary><strong>Technical details of P2 selection</strong></summary>

- The default full-grid null test uses `null_replicates=9`; it can exit early on structureless data. A failed structure check produces `no_admissible_solution` without spending confirmation fits.
- For each `k`, the candidate with the largest `Gap(k) = mean(log W*k) - log Wk` represents that value; the one-standard-error rule selects the smallest qualifying `k`. A selection at the top of `k_range` triggers a warning to widen the search.
- The default confirmation uses 10 subsamples of 80% of rows and 3 additional seeds for stochastic candidates. Admissibility requires coverage ≥ 0.8, at least 80% evaluable replicas, and median adjusted Rand index ≥ 0.7 on every applicable axis.
- Silhouettes within 0.02 of the best eligible candidate are tied; ties are resolved using stability, coverage, fitting cost, and candidate identifier.
- Alternatives include `SelectionPolicy(k_selection="silhouette")`, `null_scope="family"`, and `SelectionPolicy.p0()`. They **change the evidence policy** and should not be treated as equivalent defaults.
- The null-reference test described here belongs to the **compact** objective; other objectives have different selection rules.

See [Methods](docs/METHODS.md) and [Validation](docs/VALIDATION.md) for definitions, experimental protocol, and limitations.

</details>

## Algorithms and objectives

Clustelia keeps objectives and their evaluation metrics explicit. Scores are **not compared across incompatible objectives, geometries, or representations**.

| Objective | Supported families / adapters | Main evidence |
|:--|:--|:--|
| **Compact** | KMeans, MiniBatchKMeans, BIRCH, BisectingKMeans, Spectral, Ward, medoids, Gower k-prototypes | Silhouette in a common reference geometry, followed by selection/confirmation |
| **Probabilistic** | GaussianMixture, BayesianGaussianMixture | BIC for GaussianMixture or held-out likelihood as applicable |
| **Density** | DBSCAN, OPTICS, HDBSCAN (scikit-learn) | DBCV with the compatible Euclidean reference and optional backend |
| **Fuzzy** | Fuzzy C-Means | Generalized Xie–Beni with a fixed fuzzifier |

Use `adapter_capabilities()` to inspect model capabilities and optional-backend availability. Transductive adapters (Ward, Spectral, DBSCAN, OPTICS, and HDBSCAN) do not support assignment to new observations.

## Mixed data

For mixed tables, declare the roles of your variables explicitly instead of hoping the algorithm infers their meaning.

```python
from clustelia import AutoCluster, GeometrySpec, Schema

model = AutoCluster(
    schema=Schema(
        numeric=["age", "income"],
        categorical=["activity"],
        ordinal={"education": ("primary", "secondary", "tertiary")},
        binary=["exports"],            # asymmetric: 0 / 1
        identifiers=["firm_id"],
    ),
    geometry=GeometrySpec(kind="gower", scaling="none"),
    k_range=(2, 3, 4),
    random_state=42,
).fit(df)  # df is your pandas DataFrame
```

With an explicit `Schema`, every column must be declared exactly once. Without one, numeric dtypes are treated as numeric and other features as nominal; **ordinal and asymmetric binary semantics are never inferred**. For non-numeric features, the default reference geometry is Gower.

Gower uses per-variable dissimilarities on `[0, 1]`: range-normalized differences for numeric variables, matching for nominal ones, ordered-level differences for ordinal features, and asymmetric-binary comparisons that exclude joint absences. Weights can be supplied when some features should contribute more strongly.

Two default Gower candidate families are available: **medoids** (dense pairwise distances and seeded restarts) and **Gower k-prototypes** (assignable prototypes, typically lower pairwise-memory requirements). These are not interchangeable with PAM or Huang's k-prototypes.

<details>
<summary><strong>Missing values, vectorized Gower, and comparability</strong></summary>

Missing values are rejected by default. For Gower, `missing="pairwise"` compares each pair on observed features, subject to explicit comparability safeguards:

```python
geometry = GeometrySpec(
    kind="gower",
    scaling="none",
    missing="pairwise",
    incomparable="max",
    min_comparable_share=0.25,
)
```

Use `null_blocks` to preserve specified dependencies in permuted reference data. `incomparable="zero"` is a legacy compatibility option and **must not** be interpreted as evidence of similarity. Every row needs at least one observed feature; the pairwise approach assumes missingness is not informative about cluster membership.

The standard Gower backend preserves reference-stable summation. An optional `gower_backend="vectorized"` uses SciPy kernels for complete-case distances; verify comparability before relying on exact historical equivalence.

See [Convenience API](docs/CONVENIENCE_API.md) and [Development changes](docs/DEV2_CHANGES.md).

</details>

## Advanced configuration

Use `AutoCluster` to control the geometry, algorithms, and resource limits directly:

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

plan = model.plan(X)  # validates and lists candidates; does not fit
model.fit(X)
```

Use `families=(...)` to generate an automatic grid over `k_range`, **or** `candidates=(...)` for an explicit set; they are mutually exclusive. PCA may change the *fitting* space, but compact candidates are evaluated in the common reference geometry.

### Resource-aware execution

- Input data are copied and never silently imputed, modified, or densified. Invalid inputs and unsupported sample weights raise explicit errors.
- `SearchBudget` tracks every fit attempt, including errors and confirmation runs. A truncated search reports `incomplete` instead of silently weakening criteria.
- `timeout_s` is a **soft** limit checked between fits. `max_pairwise_mb` prevents excessively large *estimated* dense pairwise allocations; it is not a process-memory limit.
- The null-reference phase can use multiple worker processes through `SearchBudget(workers=...)`. This trades memory for throughput without changing the intended seeds and fit counts.
- For larger tables, CLARA-style medoids are an **opt-in approximation** (`CandidateSpec("clara", ...)`). They do not make the full workflow out-of-core.

## Saving and loading

```python
from clustelia import save, load

save(model, "model.pkl")
restored = load("model.pkl", trusted=True)
```

> [!CAUTION]
> Saved models use Python pickle and may contain the training data, including identifiers. **Only load files from trusted sources.** Deserialization can execute code; the subsequent dependency/environment compatibility check is not a security boundary. Cross-version migration is not guaranteed.

## Validation and known limitations

The P2 selection policy has been evaluated on synthetic benchmarks and **18 public OpenML datasets**, using development and reserved splits. Results and the tested environment are documented in [VALIDATION.md](docs/VALIDATION.md).

| Area | Current limitation |
|:--|:--|
| **Categorical breadth** | On some wide nominal datasets, the gap-based policy abstains even when an alternative policy finds useful partitions. Review the evidence before changing selection criteria. |
| **Time-dependent data** | Default subsampling assumes rows are suitable for independent resampling; time series, panel, and grouped designs need a different protocol. |
| **Large pairwise geometry** | Ward and dense medoids can exceed the default 256 MiB pairwise-allocation guard at a few thousand rows. CLARA mitigates part of the cost, not all metric computations. |
| **Incomplete experiments** | Unavailable optional backends, incompatible geometries, exhausted budgets, or insufficient confirmation are reported explicitly rather than replaced with another metric. |
| **Maturity** | `0.3.0a1` is an alpha; the engineering contracts and tests are not a promise of production suitability. |

The published validation documents both favorable results and failure/abstention cases. For example, P2 abstained on several categorical OpenML datasets (including dermatology, credit-approval, lymph, and mushroom); the alternative silhouette policy sometimes returned weak-to-moderate agreement with class labels. These results should not be interpreted as a general guarantee for new data.

## Documentation

| Guide | Contents |
|:--|:--|
| [Convenience API](docs/CONVENIENCE_API.md) | `cluster()`, effort presets, identifiers, reports, and vectorized Gower |
| [Methods](docs/METHODS.md) | Geometry, model families, selection metrics, and statistical decisions |
| [Validation](docs/VALIDATION.md) | Benchmark design, empirical results, and limitations |
| [Development changes](docs/DEV2_CHANGES.md) | Gower comparability, CLARA, and large-table behavior |
| [Local verification](docs/LOCAL_VERIFICATION.md) | Reproduction and verification notes |
| [Original specification](docs/SPECIFICATION_v0_1.md) | Initial design specification (Spanish) |
| [Roadmap](docs/ROADMAP.md) | Planned evolution |
| [Publishing](docs/PUBLISHING.md) | Release and packaging process |

The [`examples/`](examples/) directory contains standalone demonstrations for numeric, mixed, ordinal/binary, probabilistic, density, fuzzy, and convenience APIs.

## Development

```bash
python -m pip install -e ".[dev,all]"
python -m unittest discover -s tests -v
ruff check src tests tools examples
python tools/check_source.py src/clustelia
mypy
python -m build
python -m twine check --strict dist/*
```

The CI workflow exercises tests across supported operating systems and Python versions. Check its actual status before publishing; an informational type-check job is not equivalent to a passing type-check gate.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance and release checks, and [CHANGELOG.md](CHANGELOG.md) for version history. Bug reports and improvement proposals belong in [GitHub Issues](https://github.com/jaimevera1107/clustelia/issues).

## License

Distributed under the **BSD 3-Clause License**. See [LICENSE](LICENSE).

<sub>Clustelia logo created with ChatGPT (OpenAI).</sub>
