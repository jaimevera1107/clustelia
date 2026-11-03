"""Dependency-free static checks for the package sources.

Checks PEP 8 line length (79), trailing whitespace, tabs, a final newline,
complete type annotations on every function, docstrings on every public
module, class and function, and unused imports. It complements, and does
not replace, ruff and mypy in CI.

Usage: ``python tools/check_source.py src/clustelia``
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MAX_LINE = 79


def _line_issues(path: Path, text: str) -> list[str]:
    issues = []
    for number, line in enumerate(text.splitlines(), start=1):
        if len(line) > MAX_LINE:
            issues.append(f"{path}:{number}: line too long ({len(line)})")
        if line != line.rstrip():
            issues.append(f"{path}:{number}: trailing whitespace")
        if "\t" in line:
            issues.append(f"{path}:{number}: tab character")
    if text and not text.endswith("\n"):
        issues.append(f"{path}: missing final newline")
    return issues


def _annotation_issues(path: Path, tree: ast.Module) -> list[str]:
    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = node.args.posonlyargs + node.args.args
        arguments += node.args.kwonlyargs
        for extra in (node.args.vararg, node.args.kwarg):
            if extra is not None:
                arguments.append(extra)
        for argument in arguments:
            if argument.arg in ("self", "cls"):
                continue
            if argument.annotation is None:
                issues.append(
                    f"{path}:{node.lineno}: {node.name}() argument "
                    f"{argument.arg!r} lacks an annotation"
                )
        if node.returns is None:
            issues.append(
                f"{path}:{node.lineno}: {node.name}() lacks a return "
                "annotation"
            )
    return issues


def _docstring_issues(path: Path, tree: ast.Module) -> list[str]:
    issues = []
    if ast.get_docstring(tree) is None:
        issues.append(f"{path}: module docstring missing")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            public = not node.name.startswith("_")
            if public and ast.get_docstring(node) is None:
                issues.append(
                    f"{path}:{node.lineno}: {node.name} lacks a docstring"
                )
    return issues


def _exported_names(tree: ast.Module) -> set[str]:
    """Names listed in a literal ``__all__``."""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__"
            for t in node.targets
        ) and isinstance(node.value, (ast.List, ast.Tuple)):
            return {
                e.value for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
    return set()


def _unused_import_issues(path: Path, tree: ast.Module) -> list[str]:
    imported: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = (alias.asname or alias.name).split(".")[0]
                if name != "annotations":
                    imported[name] = node.lineno
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= _exported_names(tree)
    return [
        f"{path}:{line}: unused import {name!r}"
        for name, line in imported.items()
        if name not in used
    ]


def main(root: str) -> int:
    """Run every check under ``root`` and return the exit status."""
    issues: list[str] = []
    for path in sorted(Path(root).rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        issues += _line_issues(path, text)
        issues += _annotation_issues(path, tree)
        issues += _docstring_issues(path, tree)
        issues += _unused_import_issues(path, tree)
    for issue in issues:
        print(issue)
    print(f"{len(issues)} issue(s)")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "src/clustelia"))
