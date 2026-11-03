# Clustelia

**Clustelia** is a Python project for clustering analysis and, over time, automated clustering and model selection.

## Current scope — 0.0.0

This is an **early, functional preview**, not a completed AutoML toolkit. It provides a dependency-free utility to summarize cluster assignments (sizes, proportions, number of clusters, and noise count). Fitting clustering models, model selection and advanced stability analysis are planned, not available in this release.

### Install

```bash
pip install clustelia==0.0.0
```

### Use

```python
from clustelia import summarize_clusters

summary = summarize_clusters([0, 0, 1, -1])
print(summary.n_clusters)  # 2
print(summary.counts)      # {0: 2, 1: 1, -1: 1}
print(summary.shares)      # {0: 0.5, 1: 0.25, -1: 0.25}
```

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Releasing

PyPI publishes automatically through GitHub Actions and Trusted Publishing when a GitHub Release is **published** with a version tag matching `pyproject.toml` (e.g., `v0.0.0`). The workflow checks the version, runs tests, builds source and wheel distributions and uploads them to PyPI.

For the **first publication**, register a PyPI pending trusted publisher first:

- PyPI project: `clustelia`
- GitHub owner: `jaimevera1107`
- Repository: `clustelia`
- Workflow: `publish.yml`
- Environment: `pypi`

Then publish a GitHub Release for tag `v0.0.0`.

**Important:** A pending trusted publisher does not reserve a PyPI project name. The first successful upload creates the project. Releasing a version twice is not possible on PyPI; update the version before creating each new release.

## License

A project license has not yet been selected. Choose and add one before broad distribution if you want to grant explicit open-source reuse rights.
