"""Cohort-level summaries over meds_reader subjects: code frequency and
basic demographics. Exploratory (issue #3), not leakage-sensitive -- no
prediction_time censoring here. The leakage-safe/leaky graph-construction
paths (issues #6/#7) are separate, purpose-built functions in loader.py.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

import pandas as pd

from ehrgraph.data.loader import Event


class SubjectLike(Protocol):
    subject_id: int
    events: Iterable[Event]


class DatabaseLike(Protocol):
    def __getitem__(self, subject_id: int) -> SubjectLike: ...


def count_codes(db: DatabaseLike, subject_ids: Iterable[int], table: str) -> Counter:
    """Raw event-code frequency across a cohort, restricted to one OMOP
    source table (e.g. "condition"). One count per (subject, code) pair
    that appears -- a subject with the same code recorded twice on
    different visits counts twice, matching "how many diagnosis events",
    not "how many patients have this diagnosis" (see `patient_code_counts`
    for the latter)."""
    counts: Counter = Counter()
    for sid in subject_ids:
        subject = db[sid]
        for event in subject.events:
            if getattr(event, "table", None) == table:
                counts[event.code] += 1
    return counts


def patient_code_counts(db: DatabaseLike, subject_ids: Iterable[int], table: str) -> Counter:
    """How many distinct patients have >=1 event with this code, restricted
    to one OMOP source table. Use this (not `count_codes`) for "how common
    is this diagnosis in the cohort" -- otherwise one chronic patient with
    50 visits for the same code outweighs 50 patients with it once."""
    counts: Counter = Counter()
    for sid in subject_ids:
        subject = db[sid]
        codes_seen = {e.code for e in subject.events if getattr(e, "table", None) == table}
        counts.update(codes_seen)
    return counts


_DEMOGRAPHIC_PREFIXES = ("Gender/", "Race/", "Ethnicity/")


def demographics_for_subjects(
    db: DatabaseLike, subject_ids: Iterable[int], reference_time: dict[int, datetime]
) -> pd.DataFrame:
    """One row per subject: sex, race, ethnicity (first code seen with each
    prefix -- these are static in practice), and age at `reference_time[subject_id]`
    computed from the MEDS_BIRTH event. `reference_time` is per-subject
    (e.g. each subject's own prediction_time) rather than a single shared
    date, since age-at-prediction is what's actually relevant for a cohort
    with labels spread across many different calendar dates."""
    rows = []
    for sid in subject_ids:
        subject = db[sid]
        row = {"subject_id": sid, "sex": None, "race": None, "ethnicity": None, "birth_date": None}
        for event in subject.events:
            if event.code == "MEDS_BIRTH" or event.code.startswith("MEDS_BIRTH"):
                row["birth_date"] = event.time
            elif event.code.startswith("Gender/") and row["sex"] is None:
                row["sex"] = event.code
            elif event.code.startswith("Race/") and row["race"] is None:
                row["race"] = event.code
            elif event.code.startswith("Ethnicity/") and row["ethnicity"] is None:
                row["ethnicity"] = event.code
        ref = reference_time.get(sid)
        if row["birth_date"] is not None and ref is not None:
            row["age_years"] = (ref - row["birth_date"]).days / 365.25
        else:
            row["age_years"] = None
        rows.append(row)
    return pd.DataFrame(rows)
