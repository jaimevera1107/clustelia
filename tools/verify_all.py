"""Run the verification steps of the CI on this machine and write a report.

Usage::

    python tools/verify_all.py --install      # first run: installs .[all,dev]
    python tools/verify_all.py                # later runs
    python tools/verify_all.py --only ruff mypy
    python tools/verify_all.py --skip examples
    python tools/verify_all.py --dists dist/   # reuse an existing build

Steps (in order): install, environment, ruff, source, mypy, tests,
extras, build, twine, clean_install, equivalence, examples. A step ends as

- PASS: it ran and succeeded;
- WARN: it ran; the differences are explained in the report;
- FAIL: it ran and failed;
- MISSING: it could not run because a tool or optional package is not
  installed (this is *not* a pass: the check was not performed);
- SKIPPED: not requested.

The exit code is 1 if any step failed, 2 if none failed but some step
could not run, and 0 otherwise. The report (``verification_report.md``
by default) lists the environment and the tail of every non-passing step;
send it back when something fails. Run the script on each operating
system you care about; the GitHub CI does it for Linux, Windows and
macOS with Python 3.11 to 3.13.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRAS = {
    "hdbscan": "hdbscan",
    "skfuzzy": "scikit-fuzzy",
    "matplotlib": "matplotlib",
    "openpyxl": "openpyxl",
}
"""Importable module -> distribution of every optional dependency."""
EXAMPLE_REQUIRES = {"density": "hdbscan", "fuzzy": "skfuzzy"}
"""Examples that need an optional package (by file stem)."""
DISTRIBUTIONS = (
    "numpy", "scipy", "pandas", "scikit-learn", "threadpoolctl", "hdbscan",
    "scikit-fuzzy", "matplotlib", "openpyxl", "ruff", "mypy", "build",
    "twine", "pytest", "pytest-cov", "hypothesis",
)
TAIL_LINES = 80
BLAS_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
EQUIVALENCE_RTOL = "1e-9"


@dataclass
class Context:
    """State shared by the steps."""

    install: bool
    workdir: Path
    prebuilt: Path | None = None
    dists: list[Path] = field(default_factory=list)


@dataclass
class Result:
    """Outcome of one step."""

    name: str
    status: str
    seconds: float
    detail: str = ""


def have(module: str) -> bool:
    """Whether ``module`` can be imported in this interpreter."""
    return importlib.util.find_spec(module) is not None


def run(
    command: Sequence[str],
    cwd: Path = ROOT,
    timeout: int = 3600,
    env: dict[str, str] | None = None,
    use_src: bool = True,
) -> tuple[int, str]:
    """Run a command; return ``(exit code, combined output)``.

    ``use_src`` puts ``src`` on ``PYTHONPATH`` so the checks also work
    without installing the package.
    """
    merged = dict(os.environ)
    if use_src:
        paths = [str(ROOT / "src"), merged.get("PYTHONPATH", "")]
        merged["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
    else:
        merged.pop("PYTHONPATH", None)
    merged.update(env or {})
    try:
        done = subprocess.run(
            list(command), cwd=cwd, env=merged, capture_output=True,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout} s"
    except FileNotFoundError as error:
        return 127, str(error)
    return done.returncode, done.stdout + done.stderr


def tail(text: str, lines: int = TAIL_LINES) -> str:
    """Last ``lines`` lines of ``text``."""
    return "\n".join(text.strip().splitlines()[-lines:])


def outcome(code: int, output: str) -> tuple[str, str]:
    """Map an exit code to PASS or FAIL with the output tail."""
    return ("PASS" if code == 0 else "FAIL"), tail(output)


def current_version() -> str:
    """Package version read from ``_version.py`` without importing it."""
    text = (ROOT / "src" / "clustelia" / "_version.py").read_text()
    match = re.search(r'__version__\s*:\s*str\s*=\s*"([^"]+)"', text)
    return match.group(1) if match else "unknown"


def step_environment(ctx: Context) -> tuple[str, str]:
    """Record interpreter, platform and package versions."""
    importlib.invalidate_caches()  # packages may have just been installed
    lines = [f"python {platform.python_version()} on {platform.platform()}",
             f"clustelia {current_version()} (sources)"]
    for name in DISTRIBUTIONS:
        try:
            lines.append(f"{name} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(f"{name} NOT INSTALLED")
    return "PASS", "\n".join(lines)


def step_install(ctx: Context) -> tuple[str, str]:
    """Install the package in editable mode with every extra."""
    if not ctx.install:
        return "SKIPPED", "run with --install to pip install -e .[all,dev]"
    code, output = run([sys.executable, "-m", "pip", "install", "-e",
                        ".[all,dev]"])
    if code == 0:
        return "PASS", tail(output, 6)
    return "FAIL", tail(output)


def step_ruff(ctx: Context) -> tuple[str, str]:
    """Lint with ruff exactly as the CI does."""
    if not have("ruff"):
        return "MISSING", "pip install ruff"
    code, output = run([sys.executable, "-m", "ruff", "check", "src",
                        "tests", "tools", "examples",
                        "--output-format", "concise"])
    return outcome(code, output)


def step_source(ctx: Context) -> tuple[str, str]:
    """Project docstring and style checker on the package sources."""
    code, output = run([sys.executable, "tools/check_source.py",
                        "src/clustelia"])
    return outcome(code, output)


def step_mypy(ctx: Context) -> tuple[str, str]:
    """Static typing with the configuration of ``pyproject.toml``."""
    if not have("mypy"):
        return "MISSING", "pip install mypy"
    code, output = run([sys.executable, "-m", "mypy"])
    return outcome(code, output)


def step_tests(ctx: Context) -> tuple[str, str]:
    """Full test suite, with branch coverage when pytest-cov is present."""
    if have("pytest"):
        command = [sys.executable, "-m", "pytest", "-q", "-rs"]
        if have("pytest_cov"):
            command += ["--cov=clustelia", "--cov-branch",
                        "--cov-report=term:skip-covered"]
    else:
        command = [sys.executable, "-m", "unittest", "discover", "-s",
                   "tests"]
    code, output = run(command)
    keep = [line for line in output.splitlines()
            if line.startswith(("SKIPPED", "TOTAL", "Name"))
            or re.search(r"\d+%\s*$", line)
            or re.search(r"\d+ (passed|failed)|^Ran \d+|^OK|^FAILED", line)]
    detail = "\n".join(keep) if code == 0 else tail(output)
    return ("PASS" if code == 0 else "FAIL"), detail


def step_extras(ctx: Context) -> tuple[str, str]:
    """Optional packages whose absence makes tests and examples skip."""
    missing = [dist for module, dist in EXTRAS.items() if not have(module)]
    if missing:
        return "MISSING", ("not installed: " + ", ".join(missing)
                           + "; run pip install -e .[all] and repeat")
    return "PASS", "all optional packages are importable"


def step_build(ctx: Context) -> tuple[str, str]:
    """Build the sdist and wheel in an isolated environment."""
    if ctx.prebuilt is not None:
        ctx.dists = sorted(p for p in ctx.prebuilt.iterdir() if p.is_file())
        names = ", ".join(p.name for p in ctx.dists) or "nothing found"
        return ("PASS" if ctx.dists else "FAIL"), f"prebuilt: {names}"
    if not have("build"):
        return "MISSING", "pip install build"
    out = ctx.workdir / "dist"
    try:
        code, output = run([sys.executable, "-I", "-m", "build",
                            "--outdir", str(out)])
    finally:
        shutil.rmtree(ROOT / "build", ignore_errors=True)
        for leftover in (ROOT / "src").glob("*.egg-info"):
            shutil.rmtree(leftover, ignore_errors=True)
    ctx.dists = sorted(out.glob("*")) if code == 0 else []
    if code != 0:
        return "FAIL", tail(output)
    return "PASS", "built: " + ", ".join(p.name for p in ctx.dists)


def step_twine(ctx: Context) -> tuple[str, str]:
    """Strict metadata check of the built distributions."""
    if not have("twine"):
        return "MISSING", "pip install twine"
    if not ctx.dists:
        return "SKIPPED", "the build step produced no files"
    code, output = run([sys.executable, "-m", "twine", "check", "--strict",
                        *map(str, ctx.dists)])
    return outcome(code, output)


def venv_python(venv: Path) -> Path:
    """Interpreter of a virtual environment on any operating system."""
    scripts = "Scripts" if os.name == "nt" else "bin"
    return venv / scripts / ("python.exe" if os.name == "nt" else "python")


def step_clean_install(ctx: Context) -> tuple[str, str]:
    """Install the wheel in a fresh venv and run the tests outside the tree."""
    wheels = [p for p in ctx.dists if p.suffix == ".whl"]
    if not wheels:
        return "SKIPPED", "no wheel: the build step did not run or failed"
    venv = ctx.workdir / "venv"
    code, output = run([sys.executable, "-m", "venv", str(venv)])
    if code != 0:
        return "FAIL", tail(output)
    python = str(venv_python(venv))
    code, output = run([python, "-m", "pip", "install", "--quiet",
                        f"{wheels[0]}[all]"], use_src=False)
    if code != 0:
        return "FAIL", tail(output)
    code, output = run(
        [python, "-c", "import clustelia; print(clustelia.__version__, "
         "clustelia.__file__)"], cwd=ctx.workdir, use_src=False)
    if code != 0 or "site-packages" not in output:
        return "FAIL", "package not imported from site-packages:\n" + output
    copy = ctx.workdir / "tests_copy"
    shutil.copytree(ROOT / "tests", copy, dirs_exist_ok=True)
    code, output = run([python, "-m", "unittest", "discover", "-s",
                        str(copy)], cwd=ctx.workdir, use_src=False)
    return outcome(code, output)


def step_equivalence(ctx: Context) -> tuple[str, str]:
    """Compare a fresh capture with the stored reference (tolerance mode)."""
    reference = ROOT / "docs" / "equivalence" / (
        f"reference_{current_version()}.pkl")
    if not reference.exists():
        return "FAIL", f"missing reference {reference.name}"
    fresh = ctx.workdir / "equivalence.pkl"
    code, output = run([sys.executable, "tools/equivalence.py", "capture",
                        str(fresh)], env=BLAS_ENV)
    if code != 0:
        return "FAIL", tail(output)
    code, output = run(
        [sys.executable, "tools/equivalence.py", "compare", str(reference),
         str(fresh), "--rtol", EQUIVALENCE_RTOL], env=BLAS_ENV)
    if code == 0:
        return "PASS", tail(output, 2)
    differing = re.findall(r"^(\w+)\s+\w+\s+DIFF", output, re.MULTILINE)
    if differing and set(differing) <= {"density", "fuzzy"}:
        return "WARN", (
            "only density/fuzzy differ: expected when the optional extras "
            "of this environment differ from the one that captured the "
            "reference.\n" + tail(output, 4))
    return "FAIL", tail(output)


def step_examples(ctx: Context) -> tuple[str, str]:
    """Run every script in ``examples/``."""
    lines, failed, missing = [], False, False
    for path in sorted((ROOT / "examples").glob("*.py")):
        needs = EXAMPLE_REQUIRES.get(path.stem)
        if needs and not have(needs):
            lines.append(f"{path.name}: needs {needs} (not installed)")
            missing = True
            continue
        code, output = run([sys.executable, str(path)], timeout=900)
        lines.append(f"{path.name}: {'ok' if code == 0 else 'FAIL'}")
        if code != 0:
            failed = True
            lines.append(tail(output, 8))
    status = "FAIL" if failed else ("MISSING" if missing else "PASS")
    return status, "\n".join(lines)


STEPS: dict[str, Callable[[Context], tuple[str, str]]] = {
    "install": step_install,
    "environment": step_environment,
    "ruff": step_ruff,
    "source": step_source,
    "mypy": step_mypy,
    "tests": step_tests,
    "extras": step_extras,
    "build": step_build,
    "twine": step_twine,
    "clean_install": step_clean_install,
    "equivalence": step_equivalence,
    "examples": step_examples,
}
"""Steps in execution order."""


def write_report(path: Path, results: Sequence[Result]) -> None:
    """Write the Markdown report."""
    lines = [
        f"# Verification report: clustelia {current_version()}", "",
        time.strftime("%Y-%m-%d %H:%M:%S"), "",
        "| Step | Status | Seconds |", "|---|---|---|",
    ]
    lines += [f"| {r.name} | {r.status} | {r.seconds:.0f} |"
              for r in results]
    for r in results:
        if r.detail:
            lines += ["", f"## {r.name}: {r.status}", "", "```", r.detail,
                      "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--install", action="store_true",
                        help="pip install -e .[all,dev] first")
    parser.add_argument("--only", nargs="*", choices=list(STEPS),
                        help="run only these steps")
    parser.add_argument("--skip", nargs="*", choices=list(STEPS),
                        default=[], help="do not run these steps")
    parser.add_argument("--dists", type=Path, default=None,
                        help="use the sdist/wheel in this folder instead "
                             "of building (e.g. when pip cannot download "
                             "build requirements)")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "verification_report.md")
    args = parser.parse_args(argv)
    chosen = [n for n in STEPS if (not args.only or n in args.only)
              and n not in args.skip]
    results: list[Result] = []
    with tempfile.TemporaryDirectory() as folder:
        ctx = Context(args.install, Path(folder), args.dists)
        for position, name in enumerate(chosen, 1):
            started = time.perf_counter()
            status, detail = STEPS[name](ctx)
            seconds = time.perf_counter() - started
            results.append(Result(name, status, seconds, detail))
            print(f"[{position}/{len(chosen)}] {name:14s} {status:8s} "
                  f"{seconds:6.0f} s", flush=True)
    write_report(args.report, results)
    print(f"\nreport: {args.report}")
    if any(r.status == "FAIL" for r in results):
        return 1
    return 2 if any(r.status == "MISSING" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
