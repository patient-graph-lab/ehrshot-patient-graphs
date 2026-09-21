"""Baseline spike: reproduce Stanford's published CLMBR AUROC at full data.

Loads the precomputed CLMBR embeddings, joins them to a task's official
labels by (patient_id, prediction_time), fits a plain LogisticRegression on
the official train split, and evaluates on the official test split -- the
same protocol behind EHRSHOT's own published `clmbr` + `lr_lbfgs` numbers.

Deliberately the first thing run against real data (issue #2, PROJECT_GUIDE.md
Part 5) -- if this doesn't reproduce, everything built on these features is
suspect. Verified file formats against the real EHRSHOT_ASSETS.zip, Sep 20:
  - features/clmbr_features.pkl: dict with patient_ids, labeling_time,
    data_matrix (N x 768) arrays, aligned by row.
  - benchmark/<task>/labeled_patients.csv: patient_id, prediction_time,
    value, label_type columns.
  - splits/person_id_map.csv: split, omop_person_id -- patient-level
    train/val/test assignment (2295/2232/2212 patients).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_clmbr_features(features_path: Path) -> pd.DataFrame:
    """One row per (patient_id, prediction_time), plus a 768-dim 'embedding' column."""
    with open(features_path, "rb") as f:
        obj = pickle.load(f)
    return pd.DataFrame(
        {
            "patient_id": obj["patient_ids"],
            "prediction_time": obj["labeling_time"],
            "embedding": list(obj["data_matrix"]),
        }
    )


def load_task_labels(labeled_patients_path: Path) -> pd.DataFrame:
    """A task's official labels. Columns: patient_id, prediction_time, value."""
    df = pd.read_csv(labeled_patients_path, parse_dates=["prediction_time"])
    return df[["patient_id", "prediction_time", "value"]]


def load_official_splits(person_id_map_path: Path) -> pd.DataFrame:
    """Patient-level train/val/test assignment. Columns: omop_person_id, split."""
    return pd.read_csv(person_id_map_path)


def build_baseline_dataset(
    features_path: Path, labeled_patients_path: Path, person_id_map_path: Path
) -> pd.DataFrame:
    """Join embeddings + labels + split assignment, one row per labeled example.
    Raises if any labeled example has no matching embedding or split -- a
    silent partial join here would produce a plausible-looking but wrong
    AUROC, exactly the failure mode this whole spike exists to catch."""
    features = load_clmbr_features(features_path)
    labels = load_task_labels(labeled_patients_path)
    splits = load_official_splits(person_id_map_path)

    merged = labels.merge(features, on=["patient_id", "prediction_time"], how="inner")
    if len(merged) != len(labels):
        missing = len(labels) - len(merged)
        raise ValueError(
            f"{missing}/{len(labels)} labeled examples had no matching CLMBR "
            "embedding -- the (patient_id, prediction_time) join failed for some rows."
        )

    merged = merged.merge(
        splits.rename(columns={"omop_person_id": "patient_id"}), on="patient_id", how="left"
    )
    if merged["split"].isna().any():
        raise ValueError("Some patients have no official train/val/test split assignment.")
    return merged


#: L2 inverse-regularization strengths to search over, selected by val AUROC.
#: Verified against real data (guo_los, Sep 20): sklearn's default C=1.0 gives
#: AUROC 0.747 -- badly under-regularized for 768-dim embeddings on ~2.5k
#: training examples. Best-by-val C landed at ~0.003, test AUROC 0.815,
#: matching the published 0.81374. Regularization tuning is not optional here.
DEFAULT_C_GRID = [
    0.00001,
    0.00003,
    0.0001,
    0.0003,
    0.001,
    0.002,
    0.003,
    0.005,
    0.007,
    0.01,
    0.02,
    0.05,
    0.1,
    1.0,
]


def run_baseline_spike(
    features_path: Path,
    labeled_patients_path: Path,
    person_id_map_path: Path,
    c_grid: list[float] | None = None,
) -> dict:
    """Fit LogisticRegression on the official train split at full data (k=-1),
    selecting L2 strength C by val-set AUROC, then evaluate on the official
    test split. Returns the score, the selected C, and row counts, for
    comparison against EHRSHOT's own published number.

    Regularization tuning matters a lot here -- see DEFAULT_C_GRID.
    """
    data = build_baseline_dataset(features_path, labeled_patients_path, person_id_map_path)

    train = data[data["split"] == "train"]
    val = data[data["split"] == "val"]
    test = data[data["split"] == "test"]

    X_train = np.stack(train["embedding"].to_numpy()).astype(np.float64)
    y_train = train["value"].to_numpy()
    X_val = np.stack(val["embedding"].to_numpy()).astype(np.float64)
    y_val = val["value"].to_numpy()
    X_test = np.stack(test["embedding"].to_numpy()).astype(np.float64)
    y_test = test["value"].to_numpy()

    best_c, best_val_auroc, best_clf = None, -1.0, None
    for c in c_grid or DEFAULT_C_GRID:
        clf = LogisticRegression(max_iter=2000, C=c)
        clf.fit(X_train, y_train)
        val_auroc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        if val_auroc > best_val_auroc:
            best_c, best_val_auroc, best_clf = c, val_auroc, clf

    test_auroc = roc_auc_score(y_test, best_clf.predict_proba(X_test)[:, 1])

    return {
        "auroc": test_auroc,
        "selected_c": best_c,
        "val_auroc": best_val_auroc,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
