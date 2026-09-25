import pandas as pd
import pytest

from ehrgraph.eval.aggregate import SCHEMA, write_aggregate

ROW = {"task": "guo_los", "condition": "safe", "graph_variant": "v0", "metric": "density"}


def test_writes_schema_columns_in_order_and_fills_optional(tmp_path):
    out = tmp_path / "results" / "aggregate" / "x.csv"
    write_aggregate([ROW | {"value": 0.1}], out)
    df = pd.read_csv(out)
    assert list(df.columns) == SCHEMA
    assert df.loc[0, "value"] == pytest.approx(0.1)
    assert df["ci_low"].isna().all()


def test_refuses_patient_identifying_column(tmp_path):
    out = tmp_path / "results" / "aggregate" / "x.csv"
    with pytest.raises(ValueError, match="patient-identifying"):
        write_aggregate([ROW | {"value": 1.0, "patient_id": 123}], out)
    assert not out.exists()


def test_refuses_path_outside_results_aggregate(tmp_path):
    with pytest.raises(ValueError, match="results/aggregate"):
        write_aggregate([ROW | {"value": 1.0}], tmp_path / "results" / "per_patient" / "x.csv")


def test_refuses_unknown_and_missing_columns(tmp_path):
    out = tmp_path / "results" / "aggregate" / "x.csv"
    with pytest.raises(ValueError, match="not in the aggregate schema"):
        write_aggregate([ROW | {"value": 1.0, "extra": 2}], out)
    with pytest.raises(ValueError, match="required"):
        write_aggregate([{"task": "t", "value": 1.0}], out)


def test_refuses_non_numeric_value(tmp_path):
    out = tmp_path / "results" / "aggregate" / "x.csv"
    with pytest.raises(ValueError, match="numeric"):
        write_aggregate([ROW | {"value": "high"}], out)
