import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ehrgraph.eval.baseline import build_baseline_dataset, run_baseline_spike


def _write_synthetic_inputs(tmp_path):
    features_path = tmp_path / "clmbr_features.pkl"
    labels_path = tmp_path / "labeled_patients.csv"
    splits_path = tmp_path / "person_id_map.csv"

    with open(features_path, "wb") as f:
        pickle.dump(
            {
                "patient_ids": np.array([1, 1, 2, 3]),
                "labeling_time": np.array(
                    [
                        datetime(2020, 1, 1),
                        datetime(2020, 6, 1),
                        datetime(2020, 1, 1),
                        datetime(2020, 1, 1),
                    ]
                ),
                "data_matrix": np.zeros((4, 768)),
            },
            f,
        )

    pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "prediction_time": [datetime(2020, 1, 1)] * 3,
            "value": [True, False, True],
            "label_type": ["boolean"] * 3,
        }
    ).to_csv(labels_path, index=False)

    pd.DataFrame({"split": ["train", "train", "test"], "omop_person_id": [1, 2, 3]}).to_csv(
        splits_path, index=False
    )

    return features_path, labels_path, splits_path


def test_build_baseline_dataset_joins_correctly(tmp_path):
    features_path, labels_path, splits_path = _write_synthetic_inputs(tmp_path)

    result = build_baseline_dataset(features_path, labels_path, splits_path)

    assert len(result) == 3  # one row per labeled example, not per embedding row
    assert set(result["split"]) == {"train", "test"}
    assert result[result["patient_id"] == 3]["split"].item() == "test"


def test_build_baseline_dataset_raises_on_missing_embedding(tmp_path):
    features_path, labels_path, splits_path = _write_synthetic_inputs(tmp_path)
    # Add a labeled example with no matching (patient_id, prediction_time) in features
    labels = pd.read_csv(labels_path)
    labels.loc[len(labels)] = [99, "2099-01-01", True, "boolean"]
    labels.to_csv(labels_path, index=False)

    with pytest.raises(ValueError, match="no matching CLMBR embedding"):
        build_baseline_dataset(features_path, labels_path, splits_path)


def test_build_baseline_dataset_raises_on_missing_split(tmp_path):
    features_path, labels_path, splits_path = _write_synthetic_inputs(tmp_path)
    # Remove patient 3 from the split assignment
    splits = pd.read_csv(splits_path)
    splits[splits["omop_person_id"] != 3].to_csv(splits_path, index=False)

    with pytest.raises(ValueError, match="no official train/val/test split"):
        build_baseline_dataset(features_path, labels_path, splits_path)


def test_run_baseline_spike_selects_c_and_returns_consistent_counts(tmp_path):
    # A small but separable synthetic dataset -- real reproduction of the
    # published EHRSHOT number is verified separately against real data
    # (see PROJECT_GUIDE.md / issue #2); this just checks the val-based
    # C-selection and train/val/test plumbing actually runs end to end.
    rng = np.random.default_rng(0)
    n_per_split = 40
    patient_id, prediction_time, value, split, embeddings = [], [], [], [], []
    pid = 0
    for split_name in ["train", "val", "test"]:
        for i in range(n_per_split):
            pid += 1
            label = i % 2 == 0
            vec = rng.normal(loc=1.0 if label else -1.0, scale=0.5, size=768)
            patient_id.append(pid)
            prediction_time.append(datetime(2020, 1, 1) + timedelta(days=i))
            value.append(label)
            split.append(split_name)
            embeddings.append(vec)

    features_path = tmp_path / "clmbr_features.pkl"
    labels_path = tmp_path / "labeled_patients.csv"
    splits_path = tmp_path / "person_id_map.csv"

    with open(features_path, "wb") as f:
        pickle.dump(
            {
                "patient_ids": np.array(patient_id),
                "labeling_time": np.array(prediction_time),
                "data_matrix": np.stack(embeddings),
            },
            f,
        )
    pd.DataFrame(
        {
            "patient_id": patient_id,
            "prediction_time": prediction_time,
            "value": value,
            "label_type": ["boolean"] * len(patient_id),
        }
    ).to_csv(labels_path, index=False)
    pd.DataFrame({"split": split, "omop_person_id": patient_id}).to_csv(splits_path, index=False)

    result = run_baseline_spike(features_path, labels_path, splits_path)

    assert result["n_train"] == result["n_val"] == result["n_test"] == n_per_split
    assert result["selected_c"] in run_baseline_spike.__globals__["DEFAULT_C_GRID"]
    assert 0.0 <= result["auroc"] <= 1.0
    assert result["auroc"] > 0.8  # signal is strongly separable by construction
