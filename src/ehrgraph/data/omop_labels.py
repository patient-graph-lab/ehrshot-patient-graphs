"""Rebuild EHRSHOT's outcome labels from the course OMOP tables.

Definitions follow the EHRSHOT paper (Wornow et al. 2023, task appendix),
with the "inpatient" details pinned down by matching Stanford's released
labels on Oct 2 2026:

  Long Length of Stay  Each Inpatient Visit (OMOP concept 9201) that does
                       not end on its start day. Prediction time 23:59 on
                       the admission day. Positive if it lasts >= 7 days.
                       Matches 6,930 of Stanford's 6,995 stays (99.1%);
                       outcomes agree on 99.84% of them.

  New diagnosis        Each Inpatient Visit, not same-day. Prediction time
  (e.g. Hypertension)  23:59 on the discharge day. Skipped if the patient
                       already has the diagnosis (the root concept or any
                       descendant) before then. Positive if it first appears
                       within 365 days. No minimum-follow-up rule (adding
                       one made the match worse). Hypertension: matches 3,499
                       of Stanford's 3,764 discharges (93%); outcomes agree
                       on 99.4%. The rest are 139 already-diagnosed
                       discharges Stanford keeps despite its own rule (we
                       follow the rule) and ~115 same-day visits Stanford
                       includes only sometimes.

One label row per (patient, prediction time); when several visits share
one, the visit that ends last is kept.
"""

from __future__ import annotations

import pandas as pd

from ehrgraph.data.omop import OmopTables

INPATIENT_VISIT = 9201
#: SNOMED 59621000 "Essential hypertension" -- EHRSHOT's Hypertension root.
ESSENTIAL_HYPERTENSION = 320128
END_OF_DAY = pd.Timedelta(hours=23, minutes=59)


def inpatient_stays(tables: OmopTables) -> pd.DataFrame:
    """Inpatient visits that span at least two calendar days."""
    v = tables.visits
    v = v[(v["visit_concept_id"] == INPATIENT_VISIT)].dropna(
        subset=["visit_start_DATETIME", "visit_end_DATETIME"]
    )
    return v[v["visit_end_DATETIME"].dt.normalize() > v["visit_start_DATETIME"].dt.normalize()]


def _one_row_per_time(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values("visit_end_DATETIME")
        .drop_duplicates(["patient_id", "prediction_time"], keep="last")
        .sort_values(["patient_id", "prediction_time"])
        .reset_index(drop=True)[["patient_id", "prediction_time", "value"]]
    )


def long_los_labels(tables: OmopTables, threshold_days: int = 7) -> pd.DataFrame:
    stays = inpatient_stays(tables).rename(columns={"person_id": "patient_id"})
    stays = stays.assign(
        prediction_time=stays["visit_start_DATETIME"].dt.normalize() + END_OF_DAY,
        value=(stays["visit_end_DATETIME"] - stays["visit_start_DATETIME"])
        >= pd.Timedelta(days=threshold_days),
    )
    return _one_row_per_time(stays)


def new_diagnosis_labels(
    tables: OmopTables, root_concept_id: int, horizon_days: int = 365
) -> pd.DataFrame:
    concepts = tables.descendants([root_concept_id])
    dx = tables.events_table("condition")
    dx = dx[dx["concept_id"].isin(concepts)].dropna(subset=["time"])
    times = dx.groupby("person_id")["time"].apply(sorted).to_dict()

    stays = inpatient_stays(tables).rename(columns={"person_id": "patient_id"})
    stays = stays.assign(prediction_time=stays["visit_end_DATETIME"].dt.normalize() + END_OF_DAY)
    horizon = pd.Timedelta(days=horizon_days)
    keep, value = [], []
    for pid, pt in zip(stays["patient_id"], stays["prediction_time"], strict=True):
        ts = times.get(pid, [])
        already = any(t < pt for t in ts)
        keep.append(not already)
        value.append(not already and any(pt < t <= pt + horizon for t in ts))
    stays = stays.assign(value=value)[keep]
    return _one_row_per_time(stays)


def rebuild_task_labels(tables: OmopTables, task: str) -> pd.DataFrame:
    """Labels for one of the project's locked tasks, in the same shape as
    Stanford's labeled_patients.csv (patient_id, prediction_time, value)."""
    if task == "guo_los":
        return long_los_labels(tables)
    if task == "new_hypertension":
        return new_diagnosis_labels(tables, ESSENTIAL_HYPERTENSION)
    raise ValueError(f"no OMOP label rebuild for task {task!r}")
