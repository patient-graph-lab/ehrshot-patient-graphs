"""Read the course's OMOP tables as if they were the meds_reader database.

Every graph builder in this package (`build_node_vectors`, the censoring
functions in `loader`, the disease-trajectory builder) reads a database
shaped like meds_reader: iterate it for patient ids, index it for a
subject whose `.events` are `Event(time, code, table, visit_id)`.
`OmopDatabase` gives the course's OMOP CSVs exactly that shape, so the
whole pipeline runs on them unchanged -- the censoring logic still lives
in one place (`loader.events_before` / `events_all`).

Two coding modes:

  "concept"  each diagnosis is its OMOP standard concept (SNOMED for
             conditions): `OMOP/<concept_id>`. ICD-9 and ICD-10 codes for
             one disease become one code, so graphs don't split by the
             2015 coding switch (notebook 08). PRIMARY.
  "source"   the original billing code: `ICD9CM/401.9`, `ICD10CM/I10`.
             Reproduces the coding-era split, for the robustness check.

Rows with no standard concept (concept_id 0, about 0.1% of conditions)
are dropped in "concept" mode rather than all collapsing into one fake
shared code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ehrgraph.data.loader import Event

#: table name (as used everywhere else in the package) -> CSV, id, time,
#: concept and source-value columns
TABLES = {
    "condition": (
        "condition_occurrence.csv",
        "condition_start_DATETIME",
        "condition_concept_id",
        "condition_source_value",
    ),
    "drug_exposure": (
        "drug_exposure.csv",
        "drug_exposure_start_DATETIME",
        "drug_concept_id",
        "drug_source_value",
    ),
    "procedure": (
        "procedure_occurrence.csv",
        "procedure_DATETIME",
        "procedure_concept_id",
        "procedure_source_value",
    ),
}

#: The US switched from ICD-9-CM to ICD-10-CM billing codes on this date.
ICD10_SWITCH = pd.Timestamp("2015-10-01")
_ICD9_NUMERIC = re.compile(r"^\d{3}")
_ICD_LETTER = re.compile(r"^[A-Z]\d")


def icd_vocabulary(source_value: str, when: pd.Timestamp) -> str:
    """ICD9CM or ICD10CM for a diagnosis source value. Numeric codes are
    ICD-9; letter codes are ICD-10, except ICD-9's V/E supplementary codes,
    which share their first letter with ICD-10 chapters and are told apart
    by date."""
    if _ICD9_NUMERIC.match(source_value):
        return "ICD9CM"
    if _ICD_LETTER.match(source_value):
        if source_value[0] in "VE" and when < ICD10_SWITCH:
            return "ICD9CM"
        return "ICD10CM"
    return "OTHER"


class OmopTables:
    """Lazy access to the course OMOP CSVs in one folder."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _read(self, name: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self.root / name, **kwargs)

    @cached_property
    def person(self) -> pd.DataFrame:
        return self._read("person.csv", usecols=["person_id", "year_of_birth", "gender_concept_id"])

    @cached_property
    def visits(self) -> pd.DataFrame:
        return self._read(
            "visit_occurrence.csv",
            usecols=[
                "visit_occurrence_id",
                "person_id",
                "visit_concept_id",
                "visit_start_DATETIME",
                "visit_end_DATETIME",
            ],
            parse_dates=["visit_start_DATETIME", "visit_end_DATETIME"],
        )

    def events_table(self, table: str) -> pd.DataFrame:
        csv, time_col, concept_col, source_col = TABLES[table]
        df = self._read(
            csv,
            usecols=["person_id", "visit_occurrence_id", time_col, concept_col, source_col],
            parse_dates=[time_col],
            dtype={source_col: "string"},
        )
        return df.rename(
            columns={
                time_col: "time",
                concept_col: "concept_id",
                source_col: "source_value",
                "visit_occurrence_id": "visit_id",
            }
        )

    def descendants(self, concept_ids: Iterable[int], chunksize: int = 10_000_000) -> set[int]:
        """The given standard concepts plus every descendant, from the
        pre-computed concept_ancestor table (no recursion needed)."""
        roots = set(concept_ids)
        found = set(roots)
        for chunk in pd.read_csv(
            self.root / "concept_ancestor.csv",
            usecols=["ancestor_concept_id", "descendant_concept_id"],
            chunksize=chunksize,
        ):
            found |= set(
                chunk.loc[chunk["ancestor_concept_id"].isin(roots), "descendant_concept_id"]
            )
        return found


class OmopDatabase:
    """meds_reader-shaped view of the OMOP tables. Iterating yields person
    ids; `db[person_id].events` is that person's events, time-sorted."""

    def __init__(
        self,
        tables: OmopTables,
        include: Iterable[str] = ("condition",),
        codes: str = "concept",
    ):
        if codes not in ("concept", "source"):
            raise ValueError("codes must be 'concept' or 'source'")
        self.codes = codes
        self._person_ids = [int(p) for p in tables.person["person_id"]]
        frames = []
        for table in include:
            df = tables.events_table(table).dropna(subset=["time"])
            if codes == "concept":
                df = df[df["concept_id"] > 0]
                df["code"] = "OMOP/" + df["concept_id"].astype("int64").astype(str)
            else:
                df = df.dropna(subset=["source_value"])
                vocab = [
                    icd_vocabulary(s, t) if table == "condition" else table.upper()
                    for s, t in zip(df["source_value"], df["time"], strict=True)
                ]
                df["code"] = pd.Series(vocab, index=df.index) + "/" + df["source_value"]
            df["table"] = table
            frames.append(df[["person_id", "time", "code", "table", "visit_id"]])
        events = pd.concat(frames).sort_values(["person_id", "time"])
        self._events: dict[int, list[Event]] = {
            int(pid): [
                Event(time=t, code=c, table=tb, visit_id=v)
                for t, c, tb, v in zip(g["time"], g["code"], g["table"], g["visit_id"], strict=True)
            ]
            for pid, g in events.groupby("person_id", sort=False)
        }

    def __iter__(self):
        return iter(self._person_ids)

    def __getitem__(self, person_id):
        return SimpleNamespace(subject_id=person_id, events=self._events.get(int(person_id), []))


def concept_names(codes: Iterable[str], concept_csv: Path | None) -> dict[str, str]:
    """`OMOP/<id>` -> readable concept name, when a concept table is
    available (vocabulary only, no patient data). Codes without a match,
    or with no table at all, map to themselves."""
    codes = list(codes)
    names = {c: c for c in codes}
    if concept_csv is None:
        return names
    ids = {int(c.split("/", 1)[1]): c for c in codes if c.startswith("OMOP/")}
    for chunk in pd.read_csv(
        concept_csv,
        usecols=["concept_id", "concept_name"],
        chunksize=2_000_000,
        dtype={"concept_name": "string"},
    ):
        hit = chunk[chunk["concept_id"].isin(ids)]
        for cid, name in zip(hit["concept_id"], hit["concept_name"], strict=True):
            names[ids[int(cid)]] = name
    return names
