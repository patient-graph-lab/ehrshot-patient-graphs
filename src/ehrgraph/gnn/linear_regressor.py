"""Linear baselines on the node features (no graph neural network).

A linear model reads each stay's own feature vector and never looks at the links.
The network therefore only matters through the graph-derived blocks (`n2v`,
`structure`, `rwpe`): those are how "honest vs cheating network" reaches a
linear model, which is what makes it a useful baseline for the GNNs.

Protocol (the same for every feature set, so the comparisons are fair):

  * Patient-grouped, outcome-stratified 60 / 20 / 20 train / validation / test
    splits. A patient's stays never straddle two sets (about a third of links
    join two stays of one patient).
  * Columns are standardised with the TRAIN stays' mean and std only (the same
    thing `node_features.standardize` does, done on arrays here).
  * The regularisation strength C is chosen on the validation stays; the test
    stays are scored once, with the model fitted on the training stays.
  * Everything is repeated over many split seeds. Results are reported as mean
    and spread across seeds. The test sets of different seeds overlap, so the
    spread describes split-to-split variation, not independent trials.

`model="logistic"` is L2-regularised logistic regression (the usual linear model
for a yes/no outcome). `model="ridge"` is linear regression on the 0/1 outcome
(a linear probability model, Ridge with alpha = 1 / C); only the ranking of its
scores matters for AUROC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
#: Calendar-year bands for year-adjusted reporting (the outcome rate for new high
#: blood pressure falls about fourfold across years; the 2015 coding switch is
#: the boundary between the first two bands).
YEAR_BANDS = {"2007-2014": (2007, 2014), "2015-2018": (2015, 2018), "2019-2023": (2019, 2023)}


def make_splits(y: np.ndarray, groups: np.ndarray, seed: int = 0, n_folds: int = 5) -> dict:
    """Train / val / test masks (60 / 20 / 20 for n_folds=5), stratified on the
    outcome and grouped by patient (`groups`: one integer per patient)."""
    folds = np.full(len(y), -1)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, idx) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        folds[idx] = f
    test, val = folds == 0, folds == 1
    sp = {"train": ~(test | val), "val": val, "test": test}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(groups[sp[a]]) & set(groups[sp[b]]), f"a patient is in both {a} and {b}"
    return sp


def _standardize(X: np.ndarray, scale_mask: np.ndarray, train: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).copy()
    cols = np.flatnonzero(scale_mask)
    if len(cols):
        ref = X[train][:, cols]
        mu, sd = ref.mean(0), ref.std(0)
        sd[sd < 1e-8] = 1.0
        X[:, cols] = (X[:, cols] - mu) / sd
    return X


def _fit(kind: str, C: float, X, y):
    if kind == "logistic":
        m = LogisticRegression(C=C, max_iter=3000)
    elif kind == "ridge":
        m = Ridge(alpha=1.0 / C)
    else:
        raise ValueError("model must be 'logistic' or 'ridge'")
    return m.fit(X, y)


def _score(m, X) -> np.ndarray:
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict(X)


def evaluate_arrays(X, scale_mask, y, groups, seed: int = 0, model: str = "logistic",
                    c_grid=C_GRID, year: np.ndarray | None = None) -> dict:
    """One split seed: choose C on validation, score the test stays. If `year`
    (calendar year per stay) is given, also the test AUROC within each year band
    (blank where a band has fewer than 10 positive or 10 negative test stays)."""
    sp = make_splits(y, groups, seed)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    Xs = _standardize(X, scale_mask, tr)
    best_auc, best_c, best_m = -1.0, None, None
    for C in c_grid:
        m = _fit(model, C, Xs[tr], y[tr])
        a = roc_auc_score(y[va], _score(m, Xs[va]))
        if a > best_auc:
            best_auc, best_c, best_m = a, C, m
    s = _score(best_m, Xs[te])
    out = {"seed": seed, "C": best_c, "val_auc": best_auc,
           "test_auc": roc_auc_score(y[te], s), "test_ap": average_precision_score(y[te], s),
           "test_prevalence": float(y[te].mean()), "n_test": int(te.sum())}
    if year is not None:
        yt, yr = y[te], year[te]
        for name, (lo, hi) in YEAR_BANDS.items():
            b = (yr >= lo) & (yr <= hi)
            pos = int(yt[b].sum())
            out[f"auc_{name}"] = (roc_auc_score(yt[b], s[b]) if pos >= 10 and b.sum() - pos >= 10 else np.nan)
    return out


def evaluate_data(data, seed: int = 0, model: str = "logistic", c_grid=C_GRID,
                  year: np.ndarray | None = None) -> dict:
    """`evaluate_arrays` on a PyG `Data` from `build_pyg_data` (reads x, y,
    patient_idx and x_scale_mask; the links are ignored)."""
    return evaluate_arrays(data.x.numpy(), data.x_scale_mask.numpy(), data.y.numpy(),
                           data.patient_idx.numpy(), seed, model, c_grid, year)


def run_probes(datasets: dict, keys: list, seeds, model: str = "logistic", year: np.ndarray | None = None,
               log=print) -> pd.DataFrame:
    """Every (graph, source, feature_set) in `keys` x every split seed."""
    rows = []
    for key in keys:
        d = datasets[key]
        for seed in seeds:
            rows.append({"graph": key[0], "source": key[1], "feature_set": key[2],
                         "n_features": int(d.x.shape[1]), **evaluate_data(d, seed, model, year=year)})
        log(f"done: {key}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, metric: str = "test_auc") -> pd.DataFrame:
    """Mean and spread of `metric` across split seeds, per configuration."""
    g = df.groupby(["graph", "source", "feature_set"])
    out = pd.DataFrame({"mean": g[metric].mean(), "sd": g[metric].std(), "n_seeds": g[metric].count(),
                        "test_ap": g["test_ap"].mean(), "n_features": g["n_features"].first()})
    return out


def paired(df: pd.DataFrame, a: tuple, b: tuple, metric: str = "test_auc") -> dict:
    """a minus b, paired by split seed (both were scored on identical splits).
    `fraction_a_higher`: the share of seeds where a beats b."""
    def pick(k):
        sel = (df.graph == k[0]) & (df.source == k[1]) & (df.feature_set == k[2])
        return df[sel].set_index("seed")[metric]
    diff = (pick(a) - pick(b)).dropna()
    return {"mean_diff": diff.mean(), "sd": diff.std(), "n_seeds": len(diff),
            "fraction_a_higher": float((diff > 0).mean())}