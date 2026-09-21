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


def cohort_for_vectorization() -> FakeDatabase:
    """3 subjects, shaped to tell apart the three dedup granularities and
    the ontology rollup -- for ehrgraph.graphs.vectorize.

    Subject 10, code A: recorded twice in visit 1 (a within-visit repeat)
    and once more in visit 2 -- per_occurrence=3, per_visit=2, per_patient=1.
    Also has code B once, and one drug_exposure event that a "condition"
    table filter must exclude.
    Subject 11: code A once, in a different visit -- exercises multiple
    subjects sharing a code.
    Subject 12: codes X and Y in two different visits, both of which roll
    up to the same ancestor code under the rollup map used in tests --
    per_visit dedup after rollup should merge them into one code with
    count 2, not two separate codes with count 1 each.
    """
    return FakeDatabase(
        {
            10: FakeSubject(
                10,
                [
                    Event(time=datetime(2020, 1, 1), code="A", table="condition", visit_id=1),
                    Event(time=datetime(2020, 1, 1), code="A", table="condition", visit_id=1),
                    Event(time=datetime(2020, 6, 1), code="A", table="condition", visit_id=2),
                    Event(time=datetime(2020, 1, 1), code="B", table="condition", visit_id=1),
                    Event(time=datetime(2020, 1, 1), code="Z", table="drug_exposure", visit_id=1),
                ],
            ),
            11: FakeSubject(
                11,
                [
                    Event(time=datetime(2021, 1, 1), code="A", table="condition", visit_id=9),
                ],
            ),
            12: FakeSubject(
                12,
                [
                    Event(time=datetime(2021, 1, 1), code="X", table="condition", visit_id=1),
                    Event(time=datetime(2021, 6, 1), code="Y", table="condition", visit_id=2),
                ],
            ),
        }
    )
