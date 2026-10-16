"""Publish executed notebooks, outputs included, to notebooks/executed/.

Run on a machine that has the data. Each notebook is executed (or an
already-executed copy is taken with --from), checked against the REAL
patient ids and prediction times on this machine, cleaned of local paths,
and only then written to notebooks/executed/. Anything that fails the
check is not written.

    uv run python scripts/publish_executed.py notebooks/02_long_stay_networks.ipynb ...
    uv run python scripts/publish_executed.py --from /tmp/02_out.ipynb notebooks/02_...ipynb

See src/ehrgraph/eval/publish.py for what is checked and why.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from ehrgraph.eval.publish import sanitize, scan_notebook  # noqa: E402

EXECUTED = REPO / "notebooks" / "executed"


def sensitive_tokens() -> set[str]:
    """Every real patient id and prediction time on this machine."""
    from ehrgraph.project import setup

    p = setup()
    tokens = {str(pid) for pid in p.tables.person["person_id"]}
    for task in ("guo_los", "new_hypertension"):
        labels = p.labels(task)
        tokens |= {t.strftime("%Y-%m-%d %H:%M") for t in labels["prediction_time"]}
    return tokens


def execute(notebook: Path, out: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=3600",
            str(notebook),
            "--output",
            str(out),
        ],
        check=True,
        cwd=REPO,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument(
        "--from",
        dest="executed_copy",
        type=Path,
        help="an already-executed copy to publish (one notebook only)",
    )
    args = parser.parse_args()
    if args.executed_copy and len(args.notebooks) != 1:
        parser.error("--from takes exactly one notebook")

    tokens = sensitive_tokens()
    EXECUTED.mkdir(parents=True, exist_ok=True)
    failed = False
    for notebook in args.notebooks:
        with tempfile.TemporaryDirectory() as tmp:
            executed = args.executed_copy or Path(tmp) / notebook.name
            if not args.executed_copy:
                print(f"executing {notebook} ...")
                execute(notebook, executed)
            nb = sanitize(json.loads(executed.read_text()))
            text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
            problems = scan_notebook(nb, sensitive=tokens, size=len(text.encode()))
        if problems:
            failed = True
            print(f"NOT published: {notebook.name}")
            for p in problems:
                print(f"  - {p}")
            continue
        (EXECUTED / notebook.name).write_text(text)
        print(f"published: notebooks/executed/{notebook.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
