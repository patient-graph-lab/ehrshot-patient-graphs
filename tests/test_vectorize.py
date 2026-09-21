import numpy as np
import pytest

from ehrgraph.graphs.vectorize import build_patient_vectors
from tests.fixtures.make_synthetic import cohort_for_vectorization


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
