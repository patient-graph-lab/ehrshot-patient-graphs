"""Compare freshly computed results/aggregate files with the committed ones.

    python scripts/compare_results.py COMMITTED_DIR FRESH_DIR

Every committed CSV must come back with the same rows and (nearly) the same
values, and every committed figure must be re-drawn. Exits 1 on any mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["task", "condition", "graph_variant", "metric", "seed"]
VALUES = ["value", "ci_low", "ci_high"]


def compare_csv(committed: Path, fresh: Path) -> list[str]:
    if not fresh.exists():
        return [f"{committed.name}: not re-created"]
    a = pd.read_csv(committed, keep_default_na=False, na_values=[""])
    b = pd.read_csv(fresh, keep_default_na=False, na_values=[""])
    keys = [k for k in KEYS if k in a.columns]
    merged = a.merge(b, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True)
    problems = [
        f"{committed.name}: row {dict(zip(keys, r, strict=True))} only in {side}"
        for side, tag in (("committed", "left_only"), ("fresh", "right_only"))
        for r in merged.loc[merged["_merge"] == tag, keys].itertuples(index=False)
    ]
    both = merged[merged["_merge"] == "both"]
    for col in VALUES:
        if f"{col}_old" not in both:
            continue
        old, new = both[f"{col}_old"].astype(float), both[f"{col}_new"].astype(float)
        bad = ~np.isclose(old, new, rtol=1e-6, atol=1e-9, equal_nan=True)
        for (_, r), o, n in zip(both[bad].iterrows(), old[bad], new[bad], strict=True):
            problems.append(
                f"{committed.name}: {r['task']}/{r['condition']}/{r['metric']} {col} {o} -> {n}"
            )
    return problems


def main(committed: Path, fresh: Path) -> int:
    problems = []
    csvs = sorted(committed.glob("*.csv"))
    for path in csvs:
        problems += compare_csv(path, fresh / path.name)
    figures = sorted(p.name for p in (committed / "figures").glob("*.png"))
    problems += [
        f"figure {f} not re-drawn" for f in figures if not (fresh / "figures" / f).exists()
    ]
    for p in problems:
        print("  MISMATCH", p)
    print(
        f"checked {len(csvs)} CSV files and {len(figures)} figures: "
        + ("ALL MATCH" if not problems else f"{len(problems)} problem(s)")
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
