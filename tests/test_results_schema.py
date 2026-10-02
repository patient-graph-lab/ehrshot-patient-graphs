"""Every committed results/aggregate/*.csv must follow docs/results-schema.md
(frozen, issue #18). Runs in CI against the committed files -- no patient
data needed -- so a malformed or patient-level file can't merge silently."""

from pathlib import Path

import pandas as pd
import pytest
from pandas._libs.parsers import STR_NA_VALUES

from ehrgraph.eval.aggregate import FORBIDDEN_COLUMNS, SCHEMA

RESULTS = Path(__file__).resolve().parent.parent / "results" / "aggregate"
FILES = sorted(RESULTS.glob("*.csv"))

TASKS = {"guo_los", "new_hypertension", "population"}
CONDITIONS = {
    "safe",
    "leaky",
    "null_model",
    "reference",
    "leaky_vs_safe",
    "shared_edges",
    "safe_only_edges",
    "leaky_only_edges",
    "raw_codes",
    "snomed_harmonized",
}


def test_there_are_results_to_check():
    assert FILES, "no CSVs found in results/aggregate/"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_csv_follows_the_frozen_schema(path):
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    assert list(df.columns) == SCHEMA
    assert not FORBIDDEN_COLUMNS & set(df.columns)
    assert set(df["task"]) <= TASKS, set(df["task"]) - TASKS
    assert set(df["condition"]) <= CONDITIONS, set(df["condition"]) - CONDITIONS
    for col in ("task", "condition", "graph_variant", "metric"):
        assert df[col].notna().all(), f"empty {col}"
        assert (
            not set(df[col].astype(str)) & STR_NA_VALUES
        ), f"{col} has a value pandas reads as missing"
    assert pd.api.types.is_numeric_dtype(df["value"])
    assert df["metric"].str.fullmatch(r"[a-z0-9_]+").all()
    # one number per (task, condition, variant, metric, seed): no per-patient rows
    key = ["task", "condition", "graph_variant", "metric", "seed"]
    assert not df.duplicated(key).any(), df[df.duplicated(key, keep=False)].head()


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_null_ensemble_rows_carry_their_interval(path):
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    ensemble = df[(df["condition"] == "null_model") & df["seed"].isna() & df["value"].notna()]
    assert ensemble[["ci_low", "ci_high"]].notna().all().all()
    assert (ensemble["ci_low"] <= ensemble["ci_high"]).all()
