# Publishing Clustelia

Releases are published to PyPI by `.github/workflows/release.yml` when a
**GitHub Release** is published. The workflow runs the full test matrix
(Linux, Windows, macOS; Python 3.11–3.13), ruff, the source check and mypy,
checks that the tag equals `v<package version>`, builds the wheel and the
sdist, runs `twine check --strict`, and uploads with PyPI **Trusted
Publishing** (OIDC). No API token or password is stored anywhere.

Publishing is irreversible: PyPI never accepts the same version twice, even
after a deletion. Publishing a GitHub Release is not a dry run.

## One-time setup (PyPI side)

In <https://pypi.org/manage/project/clustelia/settings/publishing/> the
project must have a trusted publisher with exactly these values:

| Field | Value |
|---|---|
| Owner | `jaimevera1107` |
| Repository | `clustelia` |
| Workflow | `release.yml` |
| Environment | `pypi` |

The `0.0.0` placeholder was uploaded by an older workflow named
`publish.yml`; a publisher registered for that name does **not** authorise
`release.yml`. The GitHub environment `pypi` already exists in the
repository settings.

## Each release

1. On a branch, set `src/clustelia/_version.py` and add the version section
   to `CHANGELOG.md`; open a PR, wait for green CI, merge into `main`.
2. Verify locally (optional but recommended):

   ```bash
   python -m pip install -e ".[dev,all]"
   python -m pytest
   ruff check src tests tools examples
   mypy
   python -m build && python -m twine check --strict dist/*
   ```

3. On GitHub: **Releases → Draft a new release**, tag `v<version>` (for
   example `v0.1.0`) created on `main`, title `Clustelia <version>`, paste
   the CHANGELOG section as notes, **Publish release**.
4. Follow the run in **Actions → Release**. When it ends green, check
   <https://pypi.org/project/clustelia/> and, in a fresh environment,
   `pip install clustelia==<version>`.

If the publish job fails (for example, a missing trusted publisher), fix
the cause and use **Re-run failed jobs** on the same run; the tag and the
GitHub Release can stay as they are because nothing reached PyPI.

## Version numbers

Public numbering starts at `0.1.0`. Use `0.x.y` while the API may change;
increase the minor number for new features or behaviour changes and the
patch number for fixes. Methodological changes also require updating
`SelectionPolicy.version` and regenerating the validation evidence.
