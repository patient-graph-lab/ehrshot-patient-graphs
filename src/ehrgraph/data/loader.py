"""Patient event loading and leakage-safe censoring.

The one function every graph-construction path must go through is
`events_before`. It's written as plain Python against a small Protocol,
not against meds_reader directly, so the safety-critical logic can be
tested with a synthetic fixture (tests/test_censoring.py) without needing
the real compiled meds_reader library or any real data.

Verified against meds_reader's own README (github.com/som-shahlab/meds_reader):
the class is `SubjectDatabase`, not `PatientDatabase` — see PROJECT_GUIDE.md
Module G for the full correction. `open_database` isolates that real
dependency to one place so nothing else in this package imports it directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Event:
    time: datetime
    code: str


class SubjectLike(Protocol):
    subject_id: int
    events: Iterable[Event]


def events_before(subject: SubjectLike, prediction_time: datetime) -> list[Event]:
    """The leakage-safe condition: only events strictly before prediction_time.

    Every safe-graph code path calls this. The intentionally-leaky condition
    calls `events_all` instead — two different function names, not one
    function with a flag, so the two code paths are visibly different in a
    diff or a code review rather than one line apart and easy to mix up.
    """
    return [e for e in subject.events if e.time < prediction_time]


def events_all(subject: SubjectLike) -> list[Event]:
    """The intentionally-leaky condition: full history, no censoring.
    Deliberate negative control — see PROJECT_GUIDE.md Part 1."""
    return list(subject.events)


def open_database(path: str):
    """Thin wrapper around meds_reader.SubjectDatabase. Import is local to
    this function so the rest of the package has no hard dependency on the
    compiled library — only code that actually touches real data needs it
    installed."""
    import meds_reader

    return meds_reader.SubjectDatabase(path)
