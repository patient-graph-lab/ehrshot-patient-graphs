"""Resolve raw event codes ("ICD9CM/623.8") to human-readable concept names.

The full EHRSHOT_ASSETS package ships a real OMOP concept.csv
(femr/logs/omop_dir/concept.csv, ~5.9M rows, all vocabularies used by the
dataset) -- this is the authoritative source for code -> name resolution,
the same table the course's OMOP "Find" step points at, not a third-party
guess. meds_reader event codes are "VOCAB/CODE" strings (e.g.
"ICD9CM/623.8"), which join directly onto concept.csv's
(vocabulary_id, concept_code) columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CONCEPT_COLUMNS = [
    "concept_id",
    "concept_name",
    "vocabulary_id",
    "standard_concept",
    "concept_code",
]


def load_concept_table(concept_csv_path: Path) -> pd.DataFrame:
    """The full concept.csv, narrowed to the columns resolution needs."""
    return pd.read_csv(
        concept_csv_path,
        usecols=CONCEPT_COLUMNS,
        dtype={col: "string" for col in CONCEPT_COLUMNS if col != "concept_id"}
        | {"concept_id": "int64"},
    )


def parse_meds_code(code: str) -> tuple[str, str]:
    """ "ICD9CM/623.8" -> ("ICD9CM", "623.8"). Splits on the first "/" only,
    since concept_code itself can legitimately contain "/" (e.g. some LOINC
    and NDC codes)."""
    vocabulary_id, _, concept_code = code.partition("/")
    if not _:
        raise ValueError(f"Not a 'VOCAB/CODE'-shaped event code: {code!r}")
    return vocabulary_id, concept_code


def resolve_codes(codes: pd.Series, concept_table: pd.DataFrame) -> pd.DataFrame:
    """One row per input code, joined to its concept name.

    Unmatched codes (real -- not every local/custom code has a concept.csv
    entry) get concept_name=NA rather than raising, but the caller should
    check the match rate: a silently-low join rate here would quietly turn
    a chart of diagnoses into a chart of junk labels.
    """
    codes = codes.rename("code") if codes.name != "code" else codes
    parsed = codes.drop_duplicates().to_frame()
    parsed[["vocabulary_id", "concept_code"]] = parsed["code"].apply(
        lambda c: pd.Series(parse_meds_code(c))
    )
    merged = parsed.merge(
        concept_table[["vocabulary_id", "concept_code", "concept_name", "standard_concept"]],
        on=["vocabulary_id", "concept_code"],
        how="left",
    )
    return merged[["code", "vocabulary_id", "concept_code", "concept_name", "standard_concept"]]
