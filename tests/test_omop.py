"""The course-OMOP data path: tiny hand-built CSVs, no real data."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ehrgraph.data.loader import events_before
from ehrgraph.data.omop import OmopDatabase, OmopTables, icd_vocabulary
from ehrgraph.data.omop_labels import (
    ESSENTIAL_HYPERTENSION,
    long_los_labels,
    new_diagnosis_labels,
)
from ehrgraph.netstats.task_analysis import analyze_task, task_graphs_from_labels

HTN_CHILD = 999  # stands in for e.g. "benign essential hypertension"


def _write_tables(root):
    pd.DataFrame(
        {"person_id": [1, 2, 3], "year_of_birth": [1950] * 3, "gender_concept_id": [8507] * 3}
    ).to_csv(root / "person.csv", index=False)
    visits = [
        # person, visit id, concept, start, end
        (1, 10, 9201, "2016-01-01 08:00", "2016-01-10 12:00"),  # 9 days -> long stay
        (1, 11, 9201, "2017-03-01 08:00", "2017-03-03 09:00"),  # 2 days
        (2, 20, 9201, "2016-05-05 08:00", "2016-05-05 20:00"),  # same day -> ignored
        (2, 21, 9202, "2016-06-01 08:00", "2016-06-20 08:00"),  # outpatient -> ignored
        (3, 30, 9201, "2018-02-01 08:00", "2018-02-04 08:00"),
    ]
    pd.DataFrame(
        visits,
        columns=[
            "person_id",
            "visit_occurrence_id",
            "visit_concept_id",
            "visit_start_DATETIME",
            "visit_end_DATETIME",
        ],
    ).to_csv(root / "visit_occurrence.csv", index=False)
    conditions = [
        # person, visit, concept, time, source value
        (1, 10, 4000, "2016-01-01 00:00", "R05"),  # cough, during first stay
        (1, None, HTN_CHILD, "2016-06-01 00:00", "401.1"),  # HTN 143 days after discharge
        (3, None, ESSENTIAL_HYPERTENSION, "2017-01-01 00:00", "I10"),  # HTN before the stay
        (3, 30, 4000, "2018-02-02 00:00", "R05"),
        (2, 20, 0, "2016-05-05 00:00", "XYZ"),  # no standard concept
    ]
    pd.DataFrame(
        conditions,
        columns=[
            "person_id",
            "visit_occurrence_id",
            "condition_concept_id",
            "condition_start_DATETIME",
            "condition_source_value",
        ],
    ).to_csv(root / "condition_occurrence.csv", index=False)
    pd.DataFrame(
        {
            "ancestor_concept_id": [ESSENTIAL_HYPERTENSION, ESSENTIAL_HYPERTENSION],
            "descendant_concept_id": [ESSENTIAL_HYPERTENSION, HTN_CHILD],
        }
    ).to_csv(root / "concept_ancestor.csv", index=False)
    return OmopTables(root)


def test_icd_vocabulary_by_pattern_and_date():
    assert icd_vocabulary("401.9", pd.Timestamp("2012-01-01")) == "ICD9CM"
    assert icd_vocabulary("I10", pd.Timestamp("2019-01-01")) == "ICD10CM"
    assert icd_vocabulary("V45.81", pd.Timestamp("2012-01-01")) == "ICD9CM"
    assert icd_vocabulary("V43.52", pd.Timestamp("2019-01-01")) == "ICD10CM"


def test_omop_database_is_meds_reader_shaped_and_censorable(tmp_path):
    tables = _write_tables(tmp_path)
    db = OmopDatabase(tables)
    assert sorted(db) == [1, 2, 3]
    events = db[1].events
    assert [e.code for e in events] == ["OMOP/4000", f"OMOP/{HTN_CHILD}"]
    assert all(e.table == "condition" for e in events)
    before = events_before(db[1], datetime(2016, 1, 1, 23, 59))
    assert [e.code for e in before] == ["OMOP/4000"]
    assert db[2].events == []  # its only condition has no standard concept

    source = OmopDatabase(tables, codes="source")
    assert [e.code for e in source[3].events] == ["ICD10CM/I10", "ICD10CM/R05"]
    assert [e.code for e in source[1].events][1] == "ICD9CM/401.1"


def test_long_los_labels_follow_the_ehrshot_rules(tmp_path):
    labels = long_los_labels(_write_tables(tmp_path))
    # person 2's visits are same-day or outpatient -> no labels
    assert set(labels["patient_id"]) == {1, 3}
    row = labels[(labels["patient_id"] == 1)].iloc[0]
    assert row["prediction_time"] == pd.Timestamp("2016-01-01 23:59")  # admission day
    assert bool(row["value"]) is True  # 9 days
    assert labels["value"].tolist() == [True, False, False]


def test_new_diagnosis_labels_use_descendants_and_skip_already_diagnosed(tmp_path):
    labels = new_diagnosis_labels(_write_tables(tmp_path), ESSENTIAL_HYPERTENSION)
    # person 3 was diagnosed before their only stay; person 1's 2017 stay comes
    # after their 2016 diagnosis -- both "already diagnosed", both skipped.
    # Person 1's 2016 discharge remains, positive via a CHILD concept 143 days later.
    assert len(labels) == 1
    row = labels.iloc[0]
    assert row["patient_id"] == 1
    assert row["prediction_time"] == pd.Timestamp("2016-01-10 23:59")  # discharge day
    assert bool(row["value"]) is True


def test_pipeline_runs_without_clmbr_and_skips_dirichlet(tmp_path):
    tables = _write_tables(tmp_path)
    db = OmopDatabase(tables)
    labels = long_los_labels(tables)
    tg = task_graphs_from_labels(db, labels, "guo_los", clmbr_path=None, n_null=1, k=2)
    assert tg.features is None and tg.feature_nodes == []
    rows = analyze_task(tg).rows
    metrics = {r["metric"] for r in rows}
    assert "dirichlet_energy_clmbr" not in metrics
    assert {"class_insensitive_homophily", "n_labels", "clmbr_coverage"} <= metrics
    coverage = next(r["value"] for r in rows if r["metric"] == "clmbr_coverage")
    assert coverage == 0.0


def test_setup_finds_the_data_folder_and_explains_what_is_missing(tmp_path, monkeypatch):
    from ehrgraph.project import setup

    _write_tables(tmp_path)
    monkeypatch.setenv("EHRSHOT_TRIMMED_ROOT", str(tmp_path))
    assert setup().data == tmp_path

    monkeypatch.setenv("EHRSHOT_TRIMMED_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="person.csv"):
        setup()


def test_dirichlet_uses_only_fingerprinted_nodes():
    import scipy.sparse as sp

    from ehrgraph.netstats.task_analysis import build_task_graphs

    t0 = datetime(2020, 1, 1)
    nodes = [(i, t0) for i in range(6)]
    matrix = sp.csr_matrix(
        np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 0, 1], [1, 0, 1]])
    )
    labels = {n: i % 2 for i, n in enumerate(nodes)}
    feats = np.random.default_rng(0).normal(size=(4, 3))
    tg = build_task_graphs(
        "t",
        nodes,
        labels,
        (matrix, list("abc")),
        (matrix, list("abc")),
        feats,
        n_null=1,
        k=2,
        feature_nodes=nodes[:4],
    )
    rows = analyze_task(tg).rows
    coverage = next(r["value"] for r in rows if r["metric"] == "clmbr_coverage")
    assert coverage == pytest.approx(4 / 6)
    assert any(r["metric"] == "dirichlet_energy_clmbr" for r in rows)
