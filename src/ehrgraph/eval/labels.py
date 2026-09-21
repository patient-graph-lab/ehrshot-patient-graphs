"""Summarize EHRSHOT's benchmark label files -- one row per task, across
however many tasks are actually released, not just the two locked for this
project. Used for the broad "what's in the whole dataset" pass in issue #3
before zooming into guo_los / new_hypertension specifically.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_benchmark_tasks(assets_root: Path) -> list[str]:
    """Every subdirectory of benchmark/ that actually has a labeled_patients.csv."""
    benchmark_dir = assets_root / "benchmark"
    return sorted(
        d.name
        for d in benchmark_dir.iterdir()
        if d.is_dir() and (d / "labeled_patients.csv").exists()
    )


def summarize_task(labeled_patients_path: Path) -> dict:
    """One task's shape: row/patient counts, label type, and class balance.
    `positive_rate` is only meaningful for boolean tasks -- for categorical
    tasks (the lab-value severity tasks, chexpert) it's left as None and
    `class_counts` carries the real distribution instead."""
    df = pd.read_csv(labeled_patients_path)
    label_type = df["label_type"].iloc[0]
    class_counts = df["value"].value_counts().to_dict()
    positive_rate = None
    if label_type == "boolean":
        positive_rate = float((df["value"].astype(str) == "True").mean())
    return {
        "n_rows": len(df),
        "n_patients": df["patient_id"].nunique(),
        "label_type": label_type,
        "class_counts": class_counts,
        "positive_rate": positive_rate,
    }


def summarize_all_tasks(assets_root: Path) -> pd.DataFrame:
    """One row per released benchmark task -- the "explore everything
    briefly" table before zooming into the two locked tasks."""
    rows = []
    for task in list_benchmark_tasks(assets_root):
        summary = summarize_task(assets_root / "benchmark" / task / "labeled_patients.csv")
        rows.append({"task": task, **summary})
    return pd.DataFrame(rows)
