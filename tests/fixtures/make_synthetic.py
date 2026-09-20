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
