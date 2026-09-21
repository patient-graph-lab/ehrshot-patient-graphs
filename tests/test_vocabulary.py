import pandas as pd

from ehrgraph.data.vocabulary import load_concept_table, parse_meds_code, resolve_codes


def _fake_concept_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concept_id": [1, 2, 3],
            "concept_name": ["Essential hypertension", "Diabetes mellitus", "Fracture of femur"],
            "vocabulary_id": ["ICD10CM", "ICD9CM", "SNOMED"],
            "standard_concept": [None, None, "S"],
            "concept_code": ["I10", "250.00", "71620000"],
        }
    )


def test_parse_meds_code_splits_on_first_slash():
    assert parse_meds_code("ICD9CM/623.8") == ("ICD9CM", "623.8")


def test_parse_meds_code_handles_slash_in_concept_code():
    # concept_code itself can contain "/" (e.g. some LOINC/NDC codes) --
    # only the first "/" should be treated as the separator.
    assert parse_meds_code("LOINC/1/2") == ("LOINC", "1/2")


def test_parse_meds_code_rejects_unshaped_code():
    import pytest

    with pytest.raises(ValueError):
        parse_meds_code("no_slash_here")


def test_resolve_codes_joins_real_names():
    concept_table = _fake_concept_table()
    codes = pd.Series(["ICD10CM/I10", "ICD9CM/250.00"])
    resolved = resolve_codes(codes, concept_table)
    names = dict(zip(resolved["code"], resolved["concept_name"], strict=True))
    assert names["ICD10CM/I10"] == "Essential hypertension"
    assert names["ICD9CM/250.00"] == "Diabetes mellitus"


def test_resolve_codes_flags_unmatched_codes_as_na():
    concept_table = _fake_concept_table()
    codes = pd.Series(["ICD10CM/does_not_exist"])
    resolved = resolve_codes(codes, concept_table)
    assert pd.isna(resolved.iloc[0]["concept_name"])


def test_load_concept_table_reads_expected_columns(tmp_path):
    csv_path = tmp_path / "concept.csv"
    _fake_concept_table().to_csv(csv_path, index=False)
    df = load_concept_table(csv_path)
    assert list(df.columns) == [
        "concept_id",
        "concept_name",
        "vocabulary_id",
        "standard_concept",
        "concept_code",
    ]
    assert df["concept_id"].dtype == "int64"
