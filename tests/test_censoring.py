from datetime import datetime

from ehrgraph.data.loader import events_all, events_before
from tests.fixtures.make_synthetic import subject_with_events_around


def test_events_before_excludes_boundary_and_future():
    prediction_time = datetime(2020, 12, 1)
    subject = subject_with_events_around(prediction_time)

    result = events_before(subject, prediction_time)

    codes = {e.code for e in result}
    assert codes == {"before/1", "before/2"}
    assert all(e.time < prediction_time for e in result)


def test_events_before_is_not_a_no_op():
    """The failure mode that matters most: censoring silently doing nothing
    would still produce a graph, just a wrong one, with no error anywhere."""
    prediction_time = datetime(2020, 12, 1)
    subject = subject_with_events_around(prediction_time)

    safe = events_before(subject, prediction_time)
    leaky = events_all(subject)

    assert len(safe) < len(leaky)
    assert {e.code for e in safe} != {e.code for e in leaky}


def test_events_before_property_no_future_event_ever_included():
    """Property-style check across several prediction times on the same subject."""
    subject = subject_with_events_around(datetime(2020, 12, 1))
    for cutoff in [datetime(2019, 1, 1), datetime(2020, 6, 1), datetime(2021, 6, 1)]:
        result = events_before(subject, cutoff)
        assert all(e.time < cutoff for e in result)
