"""CI check for notebooks/executed/ -- runs without any patient data.

Fails if any executed notebook has id-shaped numbers, patient-level columns,
long tables, prediction-time timestamps, or exceeds the size cap. The exact
check against real ids happens locally in scripts/publish_executed.py; this
is the backstop on every pull request.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ehrgraph.eval.publish import scan_notebook  # noqa: E402


def main() -> int:
    paths = sorted((REPO / "notebooks" / "executed").glob("*.ipynb"))
    failed = False
    for path in paths:
        problems = scan_notebook(json.loads(path.read_text()), size=path.stat().st_size)
        for p in problems:
            failed = True
            print(f"::error file={path.relative_to(REPO)}::{p}")
    print(f"checked {len(paths)} executed notebook(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
