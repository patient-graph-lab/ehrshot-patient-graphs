import pandas as pd

from ehrgraph.eval.labels import list_benchmark_tasks, summarize_all_tasks, summarize_task


def _make_task(assets_root, task, rows):
    task_dir = assets_root / "benchmark" / task
    task_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(task_dir / "labeled_patients.csv", index=False)


def test_list_benchmark_tasks_finds_only_dirs_with_labels(tmp_path):
    _make_task(
        tmp_path,
        "task_a",
        [
            {
                "patient_id": 1,
                "prediction_time": "2020-01-01",
                "label_type": "boolean",
                "value": True,
            }
        ],
    )
    (tmp_path / "benchmark" / "empty_dir").mkdir(parents=True)
    assert list_benchmark_tasks(tmp_path) == ["task_a"]


def test_summarize_task_boolean_positive_rate(tmp_path):
    _make_task(
        tmp_path,
        "task_a",
        [
            {
                "patient_id": 1,
                "prediction_time": "2020-01-01",
                "label_type": "boolean",
                "value": True,
            },
            {
                "patient_id": 2,
                "prediction_time": "2020-01-01",
                "label_type": "boolean",
                "value": False,
            },
            {
                "patient_id": 2,
                "prediction_time": "2020-02-01",
                "label_type": "boolean",
                "value": False,
            },
            {
                "patient_id": 3,
                "prediction_time": "2020-01-01",
                "label_type": "boolean",
                "value": False,
            },
        ],
    )
    summary = summarize_task(tmp_path / "benchmark" / "task_a" / "labeled_patients.csv")
    assert summary["n_rows"] == 4
    assert summary["n_patients"] == 3
    assert summary["label_type"] == "boolean"
    assert summary["positive_rate"] == 0.25


def test_summarize_task_categorical_has_no_positive_rate(tmp_path):
    _make_task(
        tmp_path,
        "task_b",
        [
            {
                "patient_id": 1,
                "prediction_time": "2020-01-01",
                "label_type": "categorical",
                "value": 0,
            },
            {
                "patient_id": 2,
                "prediction_time": "2020-01-01",
                "label_type": "categorical",
                "value": 2,
            },
        ],
    )
    summary = summarize_task(tmp_path / "benchmark" / "task_b" / "labeled_patients.csv")
    assert summary["positive_rate"] is None
    assert summary["class_counts"] == {0: 1, 2: 1}


def test_summarize_all_tasks_one_row_per_task(tmp_path):
    _make_task(
        tmp_path,
        "task_a",
        [
            {
                "patient_id": 1,
                "prediction_time": "2020-01-01",
                "label_type": "boolean",
                "value": True,
            }
        ],
    )
    _make_task(
        tmp_path,
        "task_b",
        [
            {
                "patient_id": 1,
                "prediction_time": "2020-01-01",
                "label_type": "categorical",
                "value": 1,
            }
        ],
    )
    df = summarize_all_tasks(tmp_path)
    assert sorted(df["task"]) == ["task_a", "task_b"]
    assert len(df) == 2
