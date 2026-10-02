import pandas as pd

from ehrgraph.data.vocabulary import (
    load_concept_table,
    parse_meds_code,
    resolve_codes,
    standard_concept_map,
)


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


def test_standard_concept_map_sends_icd9_and_icd10_to_one_snomed_concept(tmp_path):
    concepts = pd.DataFrame(
        {
            "concept_id": [10, 11, 20, 30],
            "concept_name": [
                "Essential hypertension",
                "Essential hypertension (9)",
                "Essential hypertension",
                "Unmapped thing",
            ],
            "vocabulary_id": ["ICD10CM", "ICD9CM", "SNOMED", "ICD10CM"],
            "standard_concept": [None, None, "S", None],
            "concept_code": ["I10", "401.9", "59621000", "Z99"],
        }
    ).astype(
        {
            "concept_name": "string",
            "vocabulary_id": "string",
            "standard_concept": "string",
            "concept_code": "string",
        }
    )
    rel = tmp_path / "concept_relationship.csv"
    pd.DataFrame(
        {
            "concept_id_1": [10, 11, 20, 10],
            "concept_id_2": [20, 20, 10, 20],
            "relationship_id": ["Maps to", "Maps to", "Mapped from", "Is a"],
            "valid_start_date": ["2000-01-01"] * 4,
            "valid_end_date": ["2099-12-31"] * 4,
            "invalid_reason": [None] * 4,
        }
    ).to_csv(rel, index=False)

    mapping = standard_concept_map(
        ["ICD10CM/I10", "ICD9CM/401.9", "ICD10CM/Z99", "LOCAL/xyz"], concepts, rel
    )
    assert mapping["ICD10CM/I10"] == "SNOMED/59621000"
    assert mapping["ICD9CM/401.9"] == "SNOMED/59621000"
    assert mapping["ICD10CM/Z99"] == "ICD10CM/Z99"  # no "Maps to" row: unchanged
    assert mapping["LOCAL/xyz"] == "LOCAL/xyz"  # not in the vocabulary at all: unchanged
