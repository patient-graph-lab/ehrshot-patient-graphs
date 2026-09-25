"""The one door results go through on their way to GitHub.

Everything that leaves this machine is a summary row in `results/aggregate/`
(DATA_ACCESS.md, device-only rule, Sep 25). This writer enforces that at the
point of writing rather than trusting each notebook to remember it:

  - the long-format schema every downstream consumer reads
    (`task, condition, graph_variant, metric, value, ci_low, ci_high, seed`
    -- provisional until issue #18 freezes it);
  - no column that could carry a patient identity (`patient_id`,
    `prediction_time`, ...), since one stray per-patient column would turn
    an "aggregate" CSV into exactly the kind of file the DUA forbids;
  - the file lands under a `results/aggregate/` directory, nowhere else;
  - no text value that pandas' default reader turns into a missing value.
    Found the hard way (Sep 25): the graph condition "null" is on pandas'
    default NA list, so `pd.read_csv` silently dropped every null-model row
    from a groupby. Use `null_model` as the condition name instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas._libs.parsers import STR_NA_VALUES

SCHEMA = ["task", "condition", "graph_variant", "metric", "value", "ci_low", "ci_high", "seed"]
REQUIRED = ["task", "condition", "graph_variant", "metric", "value"]
FORBIDDEN_COLUMNS = {
    "patient_id",
    "subject_id",
    "person_id",
    "omop_person_id",
    "prediction_time",
    "node",
    "node_id",
}


def write_aggregate(rows: pd.DataFrame | list[dict], path: Path) -> Path:
    """Validate `rows` against the schema and write them as CSV to `path`.

    Raises ValueError on a patient-identifying column, an unknown column, a
    missing required column, a non-numeric `value`, or a destination
    outside `results/aggregate/`.
    """
    df = pd.DataFrame(rows)
    path = Path(path)

    if "results" not in path.parts or "aggregate" not in path.parts:
        raise ValueError(f"{path} is not under results/aggregate/ -- refusing to write it.")

    forbidden = FORBIDDEN_COLUMNS & set(df.columns)
    if forbidden:
        raise ValueError(f"patient-identifying column(s) {sorted(forbidden)} in aggregate output.")
    unknown = set(df.columns) - set(SCHEMA)
    if unknown:
        raise ValueError(f"column(s) {sorted(unknown)} are not in the aggregate schema {SCHEMA}.")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"required column(s) {missing} missing from aggregate output.")
    if not pd.api.types.is_numeric_dtype(df["value"]):
        raise ValueError("`value` must be numeric -- one number per (metric, condition) row.")

    for col in ("task", "condition", "graph_variant", "metric"):
        as_na = sorted(set(df[col].dropna().astype(str)) & STR_NA_VALUES)
        if as_na:
            raise ValueError(
                f"{col} value(s) {as_na} would be read back as missing by pandas -- rename them "
                "(e.g. condition 'null' -> 'null_model')."
            )

    for col in SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA

    path.parent.mkdir(parents=True, exist_ok=True)
    df[SCHEMA].to_csv(path, index=False)
    return path
