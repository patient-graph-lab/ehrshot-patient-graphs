"""GraphSAGE on the patient-similarity networks, evaluated like the linear baselines.

Same protocol as `linear_probe.py`, so a GraphSAGE number can be compared with a
linear number directly:

  * the SAME patient-grouped, outcome-stratified 60 / 20 / 20 splits
    (`linear_probe.make_splits`, same split seeds);
  * columns standardised with the TRAIN stays' mean and std only;
  * the settings (`CONFIGS`) and the stopping epoch chosen on the VALIDATION
    stays; the TEST stays are scored once, at the end;
  * repeated over many split seeds, reported as mean and spread.

How the network is used (transductive, full batch)
--------------------------------------------------
The whole network is given to the model at every step: each stay's hidden vector
is built from its own features and the average of its linked stays' vectors
(`SAGEConv`, mean aggregation), over `layers` hops. Only the TRAIN stays'
outcomes are used for the loss. The features (never the outcomes) of validation
and test stays take part in message passing, which is the standard setting for
a fixed network. Because a patient's stays all sit in the same split, a test
stay is never linked to one of its own patient's training stays. A test stay can
be linked to training stays of *other* patients; that is the intended use of the
network, and their outcomes are not an input.

`torch_geometric.nn.models.GraphSAGE` does NOT use edge weights
(`supports_edge_weight` is False), so the weights stored in the `Data` are
ignored here. The network enters as present / absent links only.

Controls (the `model` column of the results)
--------------------------------------------
  * "sage"        : GraphSAGE on the saved network.
  * "mlp"         : the SAME architecture and training with every link removed.
                    "sage minus mlp" on identical splits is what the network adds.
  * "sage_nosame" : GraphSAGE with the links between two stays of the SAME
                    patient removed (about a third of the links). If a result
                    needs those links, it may be the same person's other
                    admission rather than similar patients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn.models import GraphSAGE

from ehrgraph.gnn.linear_regressor import YEAR_BANDS, _standardize, make_splits

#: Small grid, searched on the validation stays for every split seed.
CONFIGS = (
    {"hidden": 64, "layers": 2, "dropout": 0.5, "weight_decay": 1e-3, "lr": 0.01},
    {"hidden": 64, "layers": 2, "dropout": 0.2, "weight_decay": 1e-2, "lr": 0.01},
)
MODELS = ("sage", "mlp", "sage_nosame")


def pick_device(device: str | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def edges_for(data, model: str) -> torch.Tensor:
    """The links a given model may use. Both directions are already in
    `data.edge_index`; this only removes links."""
    ei = data.edge_index
    if model == "mlp":
        return ei[:, :0]
    if model == "sage_nosame":
        if getattr(data, "edge_attr", None) is None:
            raise ValueError("this Data has no same-patient flag (edge_attr)")
        same = data.edge_attr.reshape(ei.shape[1], -1)[:, 0] > 0.5
        return ei[:, ~same]
    if model == "sage":
        return ei
    raise ValueError(f"model must be one of {MODELS}")


def fit_one(X: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor, sp: dict, cfg: dict, seed: int,
            max_epochs: int = 200, patience: int = 25, min_epochs: int = 20) -> dict:
    """Train one GraphSAGE, stop on the best validation AUROC, return the scores
    of every stay at that epoch. `sp` holds boolean numpy masks."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = X.device
    tr = torch.as_tensor(sp["train"], device=dev)
    va = torch.as_tensor(sp["val"], device=dev)
    y_f = y.float()
    y_va = y[va].cpu().numpy()

    net = GraphSAGE(X.shape[1], cfg["hidden"], cfg["layers"], out_channels=1, dropout=cfg["dropout"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_auc, best_epoch, best_scores, stale = -1.0, 0, None, 0
    for epoch in range(max_epochs):
        net.train()
        opt.zero_grad()
        logit = net(X, edge_index).view(-1)
        F.binary_cross_entropy_with_logits(logit[tr], y_f[tr]).backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            s = net(X, edge_index).view(-1)
        auc = roc_auc_score(y_va, s[va].cpu().numpy())
        if auc > best_auc:
            best_auc, best_epoch, best_scores, stale = auc, epoch, s.cpu().numpy(), 0
        else:
            stale += 1
            if epoch >= min_epochs and stale >= patience:
                break
    return {"val_auc": best_auc, "epoch": best_epoch, "scores": best_scores}


def evaluate_data(data, model: str = "sage", seed: int = 0, configs=CONFIGS, year: np.ndarray | None = None,
                  subset: np.ndarray | None = None, device: str | None = None, **fit_kw) -> dict:
    """One split seed on a PyG `Data` from `build_pyg_data` (reads x, y, edge_index,
    edge_attr, patient_idx, x_scale_mask). `subset`: boolean mask of the stays to
    use at all (e.g. those with a CLMBR vector); the network is restricted to
    links among them."""
    dev = pick_device(device)
    X = data.x.numpy()
    y = data.y.numpy()
    groups = data.patient_idx.numpy()
    ei = edges_for(data, model)
    if subset is not None:
        keep = np.flatnonzero(subset)
        remap = -np.ones(len(y), dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        e = ei.numpy()
        ok = subset[e[0]] & subset[e[1]]
        ei = torch.as_tensor(remap[e[:, ok]], dtype=torch.long)
        X, y, groups = X[keep], y[keep], groups[keep]
        year = None if year is None else year[subset]

    sp = make_splits(y, groups, seed)                       # identical to the linear baselines
    Xs = _standardize(X, data.x_scale_mask.numpy(), sp["train"])
    Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.long, device=dev)
    ei = ei.to(dev)

    best = None
    for cfg in configs:
        r = fit_one(Xt, yt, ei, sp, cfg, seed, **fit_kw)
        if best is None or r["val_auc"] > best["val_auc"]:
            best = {**r, "cfg": cfg}

    te, s = sp["test"], best["scores"]
    out = {"seed": seed, "n_used": int(len(y)), "n_edges": int(ei.shape[1]),
           "hidden": best["cfg"]["hidden"], "dropout": best["cfg"]["dropout"],
           "weight_decay": best["cfg"]["weight_decay"], "epoch": best["epoch"],
           "val_auc": best["val_auc"], "test_auc": roc_auc_score(y[te], s[te]),
           "test_ap": average_precision_score(y[te], s[te]), "test_prevalence": float(y[te].mean()),
           "n_test": int(te.sum())}
    if year is not None:
        yt_, yr = y[te], year[te]
        for name, (lo, hi) in YEAR_BANDS.items():
            b = (yr >= lo) & (yr <= hi)
            pos = int(yt_[b].sum())
            out[f"auc_{name}"] = (roc_auc_score(yt_[b], s[te][b]) if pos >= 10 and b.sum() - pos >= 10 else np.nan)
    return out


def run_models(datasets: dict, runs: list, seeds, year: np.ndarray | None = None, subset: np.ndarray | None = None,
               configs=CONFIGS, device: str | None = None, log=print, **fit_kw) -> pd.DataFrame:
    """`runs`: list of (model, (graph, source, feature_set)). One row per run x split seed."""
    rows = []
    for model, key in runs:
        d = datasets[key]
        for seed in seeds:
            rows.append({"model": model, "graph": key[0], "source": key[1], "feature_set": key[2],
                         "n_features": int(d.x.shape[1]),
                         **evaluate_data(d, model, seed, configs, year, subset, device, **fit_kw)})
        log(f"done: {model} {key}")
    return pd.DataFrame(rows)


KEY = ["model", "graph", "source", "feature_set"]


def summarize(df: pd.DataFrame, metric: str = "test_auc") -> pd.DataFrame:
    g = df.groupby(KEY)
    return pd.DataFrame({"mean": g[metric].mean(), "sd": g[metric].std(), "n_seeds": g[metric].count(),
                         "test_ap": g["test_ap"].mean(), "epoch": g["epoch"].mean(), "n_features": g["n_features"].first()})


def paired(df: pd.DataFrame, a: tuple, b: tuple, metric: str = "test_auc") -> dict:
    """a minus b, paired by split seed. a, b = (model, graph, source, feature_set)."""
    def pick(k):
        sel = np.ones(len(df), dtype=bool)
        for col, v in zip(KEY, k):
            sel &= (df[col] == v).to_numpy()
        return df[sel].set_index("seed")[metric]
    diff = (pick(a) - pick(b)).dropna()
    return {"mean_diff": diff.mean(), "sd": diff.std(), "n_seeds": len(diff),
            "fraction_a_higher": float((diff > 0).mean())}
