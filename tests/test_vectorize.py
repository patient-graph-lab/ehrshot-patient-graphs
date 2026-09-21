from datetime import datetime

import numpy as np
import pytest

from ehrgraph.data.loader import Event
from ehrgraph.graphs.vectorize import build_node_vectors, build_patient_vectors
from tests.fixtures.make_synthetic import FakeDatabase, FakeSubject, cohort_for_vectorization


def _value(matrix, patient_ids, codes, patient_id, code):
    if code not in codes:
        return 0.0
    return matrix[patient_ids.index(patient_id), codes.index(code)]


def test_per_occurrence_counts_every_row():
    db = cohort_for_vectorization()
    matrix, pids, codes = build_patient_vectors(
        db, [10, 11, 12], table="condition", weighting="count", dedup="per_occurrence"
    )
    assert _value(matrix, pids, codes, 10, "A") == 3
    assert _value(matrix, pids, codes, 10, "B") == 1


def test_per_visit_collapses_within_visit_repeats():
    db = cohort_for_vectorization()
    matrix, pids, codes = build_patient_vectors(
        db, [10, 11, 12], table="condition", weighting="count", dedup="per_visit"
    )
    # subject 10's code A: 2 raw rows in visit 1 (collapses to 1) + 1 row in
    # visit 2 (its own visit) = 2 distinct visits.
    assert _value(matrix, pids, codes, 10, "A") == 2
    assert _value(matrix, pids, codes, 10, "B") == 1


def test_per_patient_collapses_to_presence():
    db = cohort_for_vectorization()
    matrix, pids, codes = build_patient_vectors(
        db, [10, 11, 12], table="condition", weighting="count", dedup="per_patient"
    )
    assert _value(matrix, pids, codes, 10, "A") == 1
    assert _value(matrix, pids, codes, 10, "B") == 1


def test_table_filter_excludes_other_relation_types():
    db = cohort_for_vectorization()
    _, _, codes = build_patient_vectors(
        db, [10], table="condition", weighting="binary", dedup="per_patient"
    )
    assert "Z" not in codes  # Z is a drug_exposure event, not condition


def test_binary_weighting_is_dedup_invariant():
    """Presence/absence doesn't care how many times or where a code
    occurred -- binary should produce the identical matrix regardless of
    dedup mode. This is the property PROJECT_GUIDE.md's "binary +
    per_patient is structurally immune to the utilization confound" claim
    relies on."""
    db = cohort_for_vectorization()
    results = [
        build_patient_vectors(db, [10, 11, 12], table="condition", weighting="binary", dedup=d)
        for d in ("per_occurrence", "per_visit", "per_patient")
    ]
    matrices = [r[0] for r in results]
    codes_lists = [r[2] for r in results]
    assert codes_lists[0] == codes_lists[1] == codes_lists[2]
    for m in matrices[1:]:
        assert np.allclose(matrices[0].toarray(), m.toarray())


def test_count_normalized_rows_sum_to_one():
    db = cohort_for_vectorization()
    matrix, pids, codes = build_patient_vectors(
        db, [10, 11, 12], table="condition", weighting="count_normalized", dedup="per_occurrence"
    )
    row_sums = np.asarray(matrix.sum(axis=1)).flatten()
    assert np.allclose(row_sums, 1.0)


def test_count_normalized_reflects_relative_frequency():
    db = cohort_for_vectorization()
    matrix, pids, codes = build_patient_vectors(
        db, [10], table="condition", weighting="count_normalized", dedup="per_occurrence"
    )
    # subject 10, per_occurrence: A=3, B=1 -> normalized A=0.75, B=0.25
    assert _value(matrix, pids, codes, 10, "A") == pytest.approx(0.75)
    assert _value(matrix, pids, codes, 10, "B") == pytest.approx(0.25)


def test_rollup_merges_codes_before_dedup():
    db = cohort_for_vectorization()
    rollup = {"X": "ANCESTOR", "Y": "ANCESTOR"}
    matrix, pids, codes = build_patient_vectors(
        db, [12], table="condition", weighting="count", dedup="per_visit", rollup=rollup
    )
    assert "X" not in codes
    assert "Y" not in codes
    assert _value(matrix, pids, codes, 12, "ANCESTOR") == 2


def test_unknown_weighting_raises():
    db = cohort_for_vectorization()
    with pytest.raises(ValueError):
        build_patient_vectors(db, [10], table="condition", weighting="nonsense")


def test_unknown_dedup_raises():
    db = cohort_for_vectorization()
    with pytest.raises(ValueError):
        build_patient_vectors(db, [10], table="condition", dedup="nonsense")


def test_codes_are_sorted_for_reproducibility():
    db = cohort_for_vectorization()
    _, _, codes = build_patient_vectors(db, [10, 11, 12], table="condition")
    assert codes == sorted(codes)


def test_matrix_shape_matches_subjects_and_codes():
    db = cohort_for_vectorization()
    matrix, pids, codes = build_patient_vectors(db, [10, 11, 12], table="condition")
    assert matrix.shape == (len(pids), len(codes))
    assert pids == [10, 11, 12]


def _cohort_with_two_admissions() -> FakeDatabase:
    """One patient with a condition before AND after a point in time
    between their two admissions -- for testing build_node_vectors'
    per-node censoring and multi-node-per-patient behavior."""
    return FakeDatabase(
        {
            20: FakeSubject(
                20,
                [
                    Event(time=datetime(2020, 1, 1), code="A", table="condition"),
                    Event(time=datetime(2020, 6, 1), code="B", table="condition"),
                ],
            )
        }
    )


def test_build_node_vectors_censors_per_node_not_per_patient():
    db = _cohort_with_two_admissions()
    # same patient, two different admissions/prediction_times
    nodes = [(20, datetime(2020, 3, 1)), (20, datetime(2020, 12, 1))]
    matrix, node_order, codes = build_node_vectors(
        db, nodes, table="condition", censor=True, weighting="binary"
    )
    assert node_order == nodes
    early_row = matrix.getrow(0).toarray().flatten()
    late_row = matrix.getrow(1).toarray().flatten()
    # as of March, only "A" (Jan) had happened yet -- "B" (June) hasn't
    assert early_row[codes.index("A")] == 1
    assert "B" not in codes or early_row[codes.index("B")] == 0
    # as of December, both had happened
    assert late_row[codes.index("A")] == 1
    assert late_row[codes.index("B")] == 1


def test_build_node_vectors_leaky_ignores_prediction_time():
    db = _cohort_with_two_admissions()
    nodes = [(20, datetime(2020, 3, 1))]  # before "B" ever happened
    matrix, _, codes = build_node_vectors(
        db, nodes, table="condition", censor=False, weighting="binary"
    )
    row = matrix.getrow(0).toarray().flatten()
    # censor=False (events_all) ignores prediction_time entirely -- "B"
    # from June shows up even though this node's prediction_time is March
    assert row[codes.index("A")] == 1
    assert row[codes.index("B")] == 1


def test_build_node_vectors_same_patient_can_appear_twice():
    db = _cohort_with_two_admissions()
    nodes = [(20, datetime(2020, 3, 1)), (20, datetime(2020, 12, 1))]
    matrix, node_order, _ = build_node_vectors(db, nodes, table="condition", censor=True)
    assert matrix.shape[0] == 2
    assert [n[0] for n in node_order] == [20, 20]
