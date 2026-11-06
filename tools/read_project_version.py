"""Print the package version (single source: ``src/clustelia/_version.py``)."""

import re
from pathlib import Path

VERSION_FILE = (
    Path(__file__).resolve().parents[1] / "src" / "clustelia" / "_version.py"
)

match = re.search(
    r'^__version__(?:: str)? = "([^"]+)"$',
    VERSION_FILE.read_text(encoding="utf-8"),
    re.MULTILINE,
)
if match is None:
    raise SystemExit(f"No __version__ found in {VERSION_FILE}")
print(match.group(1))
