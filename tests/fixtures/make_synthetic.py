"""Synthetic MEDS-shaped fixture data — no real patients, ever.

Small enough to check in and run in CI: a handful of hand-built subjects
with events straddling a prediction_time, used to test the censoring logic
in isolation from the real (compiled, DUA-gated-data-only) meds_reader
library. See ehrgraph.data.loader.
"""

from __future__ import annotations

from datetime import datetime

from ehrgraph.data.loader import Event


class FakeSubject:
    def __init__(self, subject_id: int, events: list[Event]):
        self.subject_id = subject_id
        self.events = events


def subject_with_events_around(prediction_time: datetime) -> FakeSubject:
    """One subject with events strictly before, exactly at, and after
    prediction_time — the minimal case that exercises the boundary."""
    return FakeSubject(
        subject_id=1,
        events=[
            Event(time=datetime(2020, 1, 1), code="before/1"),
            Event(time=datetime(2020, 6, 1), code="before/2"),
            Event(time=prediction_time, code="at_boundary"),
            Event(time=datetime(2021, 1, 1), code="after/1"),
        ],
    )


class FakeDatabase:
    """Minimal dict-backed stand-in for meds_reader.SubjectDatabase --
    enough surface (`db[subject_id]`) for cohort.py's DatabaseLike Protocol."""

    def __init__(self, subjects: dict[int, FakeSubject]):
        self._subjects = subjects

    def __getitem__(self, subject_id: int) -> FakeSubject:
        return self._subjects[subject_id]


def cohort_with_conditions() -> FakeDatabase:
    """3 subjects with condition/demographic events, shaped like real
    meds_reader output (table names, MEDS_BIRTH/Gender/Race codes) --
    for cohort.py's code-counting and demographics functions."""
    return FakeDatabase(
        {
            1: FakeSubject(
                1,
                [
                    Event(time=datetime(1980, 3, 1), code="MEDS_BIRTH", table="person"),
                    Event(time=datetime(1980, 3, 1), code="Gender/F", table="person"),
                    Event(time=datetime(1980, 3, 1), code="Race/5", table="person"),
                    Event(time=datetime(2020, 1, 1), code="ICD9CM/623.8", table="condition"),
                    Event(time=datetime(2020, 6, 1), code="ICD9CM/623.8", table="condition"),
                    Event(time=datetime(2020, 6, 1), code="RxNorm/1", table="drug_exposure"),
                ],
            ),
            2: FakeSubject(
                2,
                [
                    Event(time=datetime(1990, 7, 15), code="MEDS_BIRTH", table="person"),
                    Event(time=datetime(1990, 7, 15), code="Gender/M", table="person"),
                    Event(time=datetime(1990, 7, 15), code="Race/8", table="person"),
                    Event(time=datetime(2021, 1, 1), code="ICD10CM/I10", table="condition"),
                ],
            ),
            3: FakeSubject(
                3,
                [
                    Event(time=datetime(1975, 1, 1), code="MEDS_BIRTH", table="person"),
                    Event(time=datetime(2021, 1, 1), code="ICD9CM/623.8", table="condition"),
                ],
            ),
        }
    )
