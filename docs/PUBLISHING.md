# Publishing clustelia 0.3.0a1

One-time steps, in this order. Nothing else is needed.

## 1. Put the code on GitHub

```powershell
cd "C:\Users\Aorus\Documents\Libreria\clustelia"
git init
git add .
git commit -m "clustelia 0.3.0a1"
git branch -M main
git remote add origin https://github.com/jaimevera1107/clustelia.git
git push -u origin main
```

Create the empty repository `clustelia` under your account first
(github.com/new, no README, no license: they are already here). Open the
**Actions** tab: the test matrix (Linux, Windows, macOS; Python 3.11 to
3.13), ruff and mypy run on every push. Wait for green before step 3.

## 2. Register the PyPI publisher (once)

At https://pypi.org/manage/account/publishing/ add a *pending publisher*:

| Field | Value |
|---|---|
| PyPI project name | `clustelia` |
| Owner | `jaimevera1107` |
| Repository | `clustelia` |
| Workflow | `release.yml` |
| Environment | `pypi` |

In the GitHub repository, create the environment `pypi`
(Settings, Environments, New environment).

## 3. Release

GitHub, Releases, *Draft a new release*, new tag **`v0.3.0a1`** (the
workflow refuses a tag that differs from `src/clustelia/_version.py`),
*Publish release*. The workflow runs the full matrix, builds, checks the
metadata with `twine check --strict` and publishes with OIDC. Afterwards:

```
python -m pip install --pre clustelia
```

(`--pre` is needed because 0.3.0a1 is an alpha.)

## Later versions

Edit `src/clustelia/_version.py` and `CHANGELOG.md`, commit, tag
`v<version>` and publish a new release. Methodology changes also need a
new `SelectionPolicy.version` (see `CONTRIBUTING.md`).
