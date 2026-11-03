# Local verification

`tools/verify_all.py` runs on your machine the checks the GitHub CI runs,
plus a few the CI does not, and writes `verification_report.md`. It needs
internet only for `--install` and for the isolated build.

## First run

```
python -m venv .venv
.venv/bin/activate            # Windows: .venv\Scripts\activate
python tools/verify_all.py --install
```

`--install` runs `pip install -e ".[all,dev]"`: hdbscan, scikit-fuzzy,
matplotlib, openpyxl, ruff, mypy, pytest, pytest-cov, hypothesis, build and
twine. Later runs do not need the flag.

## Steps

| Step | What it checks | Needs |
|---|---|---|
| `install` | with `--install`: `pip install -e .[all,dev]` | internet |
| `environment` | Python, OS and package versions, recorded after the install | nothing |
| `ruff` | `ruff check src tests tools examples`, as in the CI | ruff |
| `source` | docstring and style rules of `tools/check_source.py` | nothing |
| `mypy` | strict typing configured in `pyproject.toml` | mypy |
| `tests` | whole suite; with pytest-cov also branch coverage and skip reasons | pytest, pytest-cov |
| `extras` | that hdbscan, scikit-fuzzy, matplotlib and openpyxl import, so the 10 skipped tests actually run | extras |
| `build` | sdist and wheel in an isolated environment | build, internet |
| `twine` | `twine check --strict` on both files | twine |
| `clean_install` | wheel installed in a fresh venv, tests run outside the source tree | internet |
| `equivalence` | fresh capture against `docs/equivalence/reference_<version>.pkl` with `--rtol 1e-9` | nothing |
| `examples` | every script in `examples/` | extras for `density` and `fuzzy` |

The report lists the detail of every step (skip reasons and coverage
under `tests`, built files under `build`, per-example results).

A step ends as **PASS**, **WARN** (ran, differences explained in the
report), **FAIL**, **MISSING** (could not run: tool or package absent, so
the check was *not* performed) or **SKIPPED**. Exit code: 1 if anything
failed, 2 if nothing failed but something could not run, 0 otherwise.

Useful options:

```
python tools/verify_all.py --only ruff mypy      # just these steps
python tools/verify_all.py --skip examples       # everything but these
python tools/verify_all.py --dists dist/         # reuse an existing build
```

## How to read the results

- **tests**: with all extras installed the skipped count should drop from
  10 to the number listed under `SKIPPED` reasons that are unrelated to
  extras (ideally 0). Any other skip reason deserves a look.
- **tests, coverage**: the last published figure is 93% branch coverage
  at 0.3.0.dev0 (135 tests). The report shows the current `TOTAL`.
- **equivalence**: `WARN` with only `density`/`fuzzy` differing is
  expected when your extras differ from the environment that captured the
  reference. Any other difference is a real change in behaviour.
- **clean_install**: confirms the package imports from `site-packages`,
  not from `src`, and that tests pass with the files that would ship.

## Other operating systems and Python versions

Run the same command on each machine or environment (Windows, macOS,
Python 3.11, 3.12, 3.13). The CI matrix in `.github/workflows/tests.yml`
does the same on every push; its green status is the reference for
publishing.

## What to send back

If a step fails, send `verification_report.md`. It contains the
environment and the tail of every step that did not pass.
