from datetime import datetime, timedelta

import pytest

from ehrgraph.data.loader import Event
from ehrgraph.graphs.trajectory import (
    bonferroni_significant,
    build_population_trajectory_graph,
    build_trajectory_graph,
    compute_relative_risks,
    first_occurrence_times,
    symmetrize_max,
)
from tests.fixtures.make_synthetic import FakeDatabase, FakeSubject


def _cohort_with_a_leading_to_b() -> FakeDatabase:
    """100 patients: 10 have A, and 8 of those get B within 30 days
    afterward -- a strong signal (80% conditional rate). Only 2 of the
    other 90 patients (who never have A) have B at all, so B's
    population-wide baseline rate stays low (10/100 = 10%) -- this
    separation is what a real relative-risk elevation needs; if B were
    already common in the general population, an 80% post-A rate
    wouldn't look elevated at all relative to that baseline."""
    subjects = {}
    for i in range(8):
        subjects[i] = FakeSubject(
            i,
            [
                Event(time=datetime(2020, 1, 1), code="A", table="condition"),
                Event(time=datetime(2020, 1, 15), code="B", table="condition"),
            ],
        )
    for i in range(8, 10):
        subjects[i] = FakeSubject(
            i, [Event(time=datetime(2020, 1, 1), code="A", table="condition")]
        )
    for i in range(10, 12):
        subjects[i] = FakeSubject(
            i, [Event(time=datetime(2020, 1, 1), code="B", table="condition")]
        )
    for i in range(12, 100):
        subjects[i] = FakeSubject(i, [])
    return FakeDatabase(subjects)


def test_first_occurrence_times_keeps_earliest_only():
    db = FakeDatabase(
        {
            1: FakeSubject(
                1,
                [
                    Event(time=datetime(2020, 6, 1), code="A", table="condition"),
                    Event(time=datetime(2020, 1, 1), code="A", table="condition"),  # earlier
                ],
            )
        }
    )
    result = first_occurrence_times(db, [1], table="condition", codes_of_interest={"A"})
    assert result[1]["A"] == datetime(2020, 1, 1)


def test_first_occurrence_times_ignores_other_tables_and_codes():
    db = FakeDatabase(
        {
            1: FakeSubject(
                1,
                [
                    Event(time=datetime(2020, 1, 1), code="A", table="drug_exposure"),
                    Event(time=datetime(2020, 1, 1), code="Z", table="condition"),
                ],
            )
        }
    )
    result = first_occurrence_times(db, [1], table="condition", codes_of_interest={"A"})
    assert result[1] == {}


def test_compute_relative_risks_detects_elevated_risk():
    db = _cohort_with_a_leading_to_b()
    first_times = first_occurrence_times(
        db, list(range(100)), table="condition", codes_of_interest={"A", "B"}
    )
    rr_df = compute_relative_risks(
        first_times,
        codes=["A", "B"],
        window=timedelta(days=30),
        min_support=1,
        n_population=100,
    )
    row = rr_df[(rr_df["a"] == "A") & (rr_df["b"] == "B")].iloc[0]
    assert row["n_a"] == 10
    assert row["n_a_then_b"] == 8
    # A-conditional rate 8/10 (80%) vs. population baseline 10/100 (10%)
    # -- a clear, real elevation, not a marginal one.
    assert row["rr"] == pytest.approx(8.0)
    assert row["pvalue"] < 0.05


def test_compute_relative_risks_respects_the_time_window():
    db = FakeDatabase(
        {
            1: FakeSubject(
                1,
                [
                    Event(time=datetime(2020, 1, 1), code="A", table="condition"),
                    Event(time=datetime(2022, 1, 1), code="B", table="condition"),  # 2 years later
                ],
            )
        }
    )
    first_times = first_occurrence_times(db, [1], table="condition", codes_of_interest={"A", "B"})
    rr_df = compute_relative_risks(
        first_times, codes=["A", "B"], window=timedelta(days=30), min_support=1, n_population=1
    )
    assert rr_df.empty  # B happened, but far outside the window


def test_compute_relative_risks_excludes_b_before_a():
    db = FakeDatabase(
        {
            1: FakeSubject(
                1,
                [
                    Event(time=datetime(2020, 1, 1), code="B", table="condition"),
                    Event(time=datetime(2020, 1, 15), code="A", table="condition"),
                ],
            )
        }
    )
    first_times = first_occurrence_times(db, [1], table="condition", codes_of_interest={"A", "B"})
    rr_df = compute_relative_risks(
        first_times, codes=["A", "B"], window=timedelta(days=30), min_support=1, n_population=1
    )
    # B happened before A here, so the only valid directional row is
    # B -> A (correctly detected) -- there must be no A -> B row, since
    # that direction never actually happened for this patient.
    assert rr_df[(rr_df["a"] == "A") & (rr_df["b"] == "B")].empty
    assert not rr_df[(rr_df["a"] == "B") & (rr_df["b"] == "A")].empty


def test_compute_relative_risks_respects_min_support():
    db = _cohort_with_a_leading_to_b()
    first_times = first_occurrence_times(
        db, list(range(100)), table="condition", codes_of_interest={"A", "B"}
    )
    rr_df = compute_relative_risks(
        first_times,
        codes=["A", "B"],
        window=timedelta(days=30),
        min_support=9,
        n_population=100,
    )
    assert rr_df.empty  # only 8 patients went A-then-B, below the min_support=9 bar


def test_bonferroni_significant_is_stricter_than_raw_alpha():
    import pandas as pd

    rr_df = pd.DataFrame(
        {
            "a": ["A"],
            "b": ["B"],
            "n_a": [10],
            "n_a_then_b": [8],
            "n_b": [10],
            "rr": [2.0],
            "pvalue": [0.01],
        }
    )
    # 0.01 clears a raw alpha=0.05, but not after Bonferroni over 200*199 pairs
    assert not bonferroni_significant(rr_df, n_codes=200, alpha=0.05).shape[0]
    # with only 2 codes (2 possible ordered pairs), the correction is much
    # gentler and 0.01 should survive
    assert bonferroni_significant(rr_df, n_codes=2, alpha=0.05).shape[0] == 1


def test_build_trajectory_graph_uses_names_and_weights():
    import pandas as pd

    significant_df = pd.DataFrame(
        {
            "a": ["A"],
            "b": ["B"],
            "n_a": [10],
            "n_a_then_b": [8],
            "n_b": [10],
            "rr": [2.5],
            "pvalue": [1e-10],
        }
    )
    graph = build_trajectory_graph(
        significant_df, code_names={"A": "Condition A", "B": "Condition B"}
    )
    assert graph.has_edge("A", "B")
    assert graph["A"]["B"]["weight"] == pytest.approx(2.5)
    assert graph.nodes["A"]["name"] == "Condition A"


def test_symmetrize_max_keeps_the_larger_weight_both_directions():
    import networkx as nx

    directed = nx.DiGraph()
    directed.add_edge("A", "B", weight=2.0)
    directed.add_edge("B", "A", weight=5.0)
    undirected = symmetrize_max(directed)
    assert undirected["A"]["B"]["weight"] == pytest.approx(5.0)


def test_symmetrize_max_keeps_one_sided_edges():
    import networkx as nx

    directed = nx.DiGraph()
    directed.add_edge("A", "B", weight=3.0)
    undirected = symmetrize_max(directed)
    assert undirected["A"]["B"]["weight"] == pytest.approx(3.0)


def _two_era_cohort() -> FakeDatabase:
    """The same A -> B pattern coded two ways: patients 0-19 in "ICD9"
    codes, 20-39 in "ICD10" codes, plus 60 patients with nothing."""
    subjects = {}
    for i in range(40):
        era = "ICD9" if i < 20 else "ICD10"
        events = [
            Event(
                time=datetime(2012 if era == "ICD9" else 2019, 1, 1),
                code=f"{era}/A",
                table="condition",
            )
        ]
        if i % 20 < 16:
            events.append(
                Event(
                    time=datetime(2012 if era == "ICD9" else 2019, 2, 1),
                    code=f"{era}/B",
                    table="condition",
                )
            )
        subjects[i] = FakeSubject(i, events)
    for i in range(40, 100):
        subjects[i] = FakeSubject(i, [])
    return FakeDatabase(subjects)


def test_code_map_merges_era_specific_codes_into_one_trajectory():
    db = _two_era_cohort()
    raw, _ = build_population_trajectory_graph(db, n_codes=4, min_support=5)
    assert set(raw.edges()) == {("ICD9/A", "ICD9/B"), ("ICD10/A", "ICD10/B")}

    code_map = {"ICD9/A": "S/A", "ICD10/A": "S/A", "ICD9/B": "S/B", "ICD10/B": "S/B"}
    merged, _ = build_population_trajectory_graph(db, n_codes=4, min_support=5, code_map=code_map)
    assert set(merged.edges()) == {("S/A", "S/B")}
