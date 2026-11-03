# Preparing and publishing Clustelia 0.3.0a1

This project **already has a GitHub repository and a published `v0.0.0`
release**. Do not create a new repository, run `git init`, force-push,
move the old tag, or overwrite the already published PyPI version.
The `v0.0.0` release remains the recovery point until the next release
is explicitly approved.

## 1. Upload the prepared source in a separate branch

1. Start from an up-to-date checkout of
   `https://github.com/jaimevera1107/clustelia` with `main` restored to
   `v0.0.0`.
2. Create a branch from `main`: `git switch -c prepare/v0.3.0a1`.
3. Replace **the files in that checkout**, excluding the `.git` directory,
   with the contents of the `clustelia/` directory in the prepared ZIP.
   In particular, delete the old `.github/workflows/publish.yml` and
   `.github/workflows/ci.yml` (the new workflows are `tests.yml` and
   `release.yml`). `git add -A` must record these removals.
4. Review `git status --short` and commit the changes:

   ```bash
   git add -A
   git commit -m "Prepare Clustelia 0.3.0a1 alpha"
   git push -u origin prepare/v0.3.0a1
   ```

5. Open a pull request targeting `main`. Do not merge it until the
   cross-platform CI checks pass and the changes are reviewed.

## 2. Run verification locally

With Python 3.11–3.13 and a dedicated virtual environment:

```bash
python -m pip install -e ".[dev,all]"
python -m unittest discover -s tests -v
ruff check src tests tools examples
python tools/check_source.py src/clustelia
mypy
python -m build
python -m twine check --strict dist/*
```

`tools/verify_all.py --install` provides additional local environment,
equivalence and clean-install checks. Some optional backends are pinned;
if an extra fails to install, report it rather than hiding a failing job.
Pickle references used by equivalence must be **trusted**.

The workflows are:

- `.github/workflows/tests.yml`: checks each push and PR;
- `.github/workflows/release.yml`: on a **published GitHub release**, it
  runs tests, quality gates and package checks, then attempts to publish
  to PyPI. Publishing a GitHub release is *not* a dry run.

## 3. Review PyPI trusted publishing before making a release

The initial `v0.0.0` used `.github/workflows/publish.yml`. The new
workflow is named **`release.yml`**. In the PyPI project's Trusted
Publishing settings, verify the publisher points to:

| Field | Value |
|---|---|
| Project | `clustelia` |
| GitHub owner | `jaimevera1107` |
| GitHub repository | `clustelia` |
| Workflow | `release.yml` |
| Environment | `pypi` |

Configure the GitHub environment `pypi` if it does not already exist.
Never publish the same version twice to PyPI; it will be rejected.

## 4. Publish the alpha only after acceptance

Confirm that `src/clustelia/_version.py` contains `0.3.0a1`, CI is green,
and there are no outstanding issues. Once merged into `main`, create
and publish a new GitHub release with the tag **`v0.3.0a1`** targeting
`main`. The release workflow checks the tag against the package version,
builds the wheel and sdist, and uses PyPI OIDC Trusted Publishing.

Install published alphas with `python -m pip install --pre clustelia`.
Never rewrite or delete `v0.0.0` or replace its tag.

## 5. Future versions

Change `src/clustelia/_version.py` and `CHANGELOG.md`, run validation,
merge a reviewed PR, then publish a distinct tag `v<version>`.
Methodological changes also require updating
`SelectionPolicy.version` and regenerating validation evidence.
