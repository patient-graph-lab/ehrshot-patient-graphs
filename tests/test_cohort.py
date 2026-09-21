from datetime import datetime

import pandas as pd

from ehrgraph.data.cohort import count_codes, demographics_for_subjects, patient_code_counts
from tests.fixtures.make_synthetic import cohort_with_conditions


def test_count_codes_counts_events_not_patients():
    db = cohort_with_conditions()
    counts = count_codes(db, [1, 2, 3], table="condition")
    # subject 1 has ICD9CM/623.8 twice, subject 3 once -> 3 events total
    assert counts["ICD9CM/623.8"] == 3
    assert counts["ICD10CM/I10"] == 1


def test_count_codes_ignores_other_tables():
    db = cohort_with_conditions()
    counts = count_codes(db, [1], table="condition")
    assert "RxNorm/1" not in counts


def test_patient_code_counts_counts_patients_not_events():
    db = cohort_with_conditions()
    counts = patient_code_counts(db, [1, 2, 3], table="condition")
    # subject 1's duplicate ICD9CM/623.8 event should count once
    assert counts["ICD9CM/623.8"] == 2
    assert counts["ICD10CM/I10"] == 1


def test_demographics_computes_age_from_meds_birth():
    db = cohort_with_conditions()
    ref = {1: datetime(2020, 3, 1), 2: datetime(2021, 7, 15), 3: datetime(2021, 1, 1)}
    df = demographics_for_subjects(db, [1, 2, 3], ref)
    row1 = df[df["subject_id"] == 1].iloc[0]
    assert row1["sex"] == "Gender/F"
    assert row1["race"] == "Race/5"
    assert abs(row1["age_years"] - 40.0) < 0.1

    row2 = df[df["subject_id"] == 2].iloc[0]
    assert abs(row2["age_years"] - 31.0) < 0.1

    row3 = df[df["subject_id"] == 3].iloc[0]
    assert pd.isna(row3["sex"])
