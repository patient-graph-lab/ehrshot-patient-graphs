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

from collections.abc import Iterable
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


def standard_concept_map(
    codes: Iterable[str],
    concept_table: pd.DataFrame,
    relationship_csv_path: Path,
    chunksize: int = 5_000_000,
) -> dict[str, str]:
    """Source code -> its OMOP standard concept, via "Maps to" in the
    shipped concept_relationship.csv. e.g. "ICD9CM/401.9" and "ICD10CM/I10"
    both -> "SNOMED/59621000" (essential hypertension).

    Why it matters (found Sep-Oct 2026, issue #14): ICD-9 (before Oct 2015)
    and ICD-10 (after) codes for the same disease share nothing as strings,
    so graphs built on raw codes split by coding era -- the disease-
    trajectory graph's two big islands were exactly the ICD-9 and ICD-10
    halves, with zero edges between them.

    Codes with no "Maps to" row, or that are already standard, map to
    themselves. With several targets (rare), the lowest concept_id wins, so
    the mapping is deterministic.
    """
    codes = sorted(set(codes))
    resolved = resolve_codes(pd.Series(codes), concept_table)
    ids = concept_table.merge(
        resolved[["code", "vocabulary_id", "concept_code"]],
        on=["vocabulary_id", "concept_code"],
    )[["code", "concept_id"]]
    wanted = set(ids["concept_id"])

    targets = []
    for chunk in pd.read_csv(
        relationship_csv_path,
        usecols=["concept_id_1", "concept_id_2", "relationship_id"],
        dtype={"concept_id_1": "int64", "concept_id_2": "int64", "relationship_id": "string"},
        chunksize=chunksize,
    ):
        hit = chunk[(chunk["relationship_id"] == "Maps to") & chunk["concept_id_1"].isin(wanted)]
        targets.append(hit[["concept_id_1", "concept_id_2"]])
    maps_to = pd.concat(targets).sort_values("concept_id_2").drop_duplicates("concept_id_1")

    standard = concept_table.set_index("concept_id")[["vocabulary_id", "concept_code"]]
    maps_to = maps_to[maps_to["concept_id_2"].isin(standard.index)]
    target_code = (
        standard.loc[maps_to["concept_id_2"], "vocabulary_id"].to_numpy()
        + "/"
        + standard.loc[maps_to["concept_id_2"], "concept_code"].to_numpy()
    )
    by_id = dict(zip(maps_to["concept_id_1"], target_code, strict=True))
    mapping = {code: code for code in codes}
    for code, concept_id in zip(ids["code"], ids["concept_id"], strict=True):
        if concept_id in by_id:
            mapping[code] = by_id[concept_id]
    return mapping
