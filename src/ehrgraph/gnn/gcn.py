"""Graph convolution (GCN) on the patient-similarity networks, evaluated like the linear, GraphSAGE and GAT runs.

Same protocol as `linear_regressor.py` / `graph_sage.py` / `graph_attention.py`, so a GCN
number can be put next to any of them directly:

  * the SAME patient-grouped, outcome-stratified 60 / 20 / 20 splits
    (`make_splits`, same split seeds);
  * columns standardised with the TRAIN stays' mean and std only;
  * the settings (`CONFIGS`) and the stopping epoch chosen on the VALIDATION
    stays; the TEST stays are scored once, at the end;
  * repeated over many split seeds, reported as mean and spread.

What GCN does differently
-------------------------
GraphSAGE averages the neighbours; GAT learns a weight per link. GCN uses a FIXED
rule: the stay itself is added as a link to itself, and every link i-j is scaled by
1 / sqrt(deg_i * deg_j) (symmetric normalisation). Nothing about the weighting is
learned. Two consequences:

  * a stay with many links gets a SMALL weight on each of them, and a link to a
    stay with many links counts for less: well-connected stays are damped;
  * with few links, the stay's own features count for more (its weight on itself
    is 1 / (deg + 1) in the unweighted case).

Details of `torch_geometric.nn.models.GCN` that matter here
  * Self-loops are added by default, so with every link removed each stay is just
    its own features passed through a linear layer: an ordinary MLP. That is the
    "mlp" control below, exact.
  * Unlike GraphSAGE and GAT, GCN DOES use edge weights (`supports_edge_weight` is
    True; verified, and all-ones weights give exactly the unweighted result). The
    stored `Data.edge_weight` is used only by the `gcn_w` model; `gcn` ignores it.
    With weights, the degree in the normalisation is the weighted degree.
  * It does not take edge features.
  * Full batch and transductive, exactly as in `graph_sage.py`: the whole network
    is visible, only TRAIN stays' outcomes enter the loss, and a patient's stays
    all sit in one split.

Models (the `model` column of the results)
  * "gcn"         : GCN on the saved network, links present / absent.
  * "gcn_w"       : the same, with each link weighted by the stored `edge_weight`.
                    "gcn_w minus gcn" is what the weights add.
  * "mlp"         : the SAME architecture and training with every link removed.
                    "gcn minus mlp" on identical splits is what the network adds.
  * "gcn_nosame"  : GCN with the links between two stays of the SAME patient
                    removed (about a third of the links).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn.models import GCN

from ehrgraph.gnn.linear_regressor import YEAR_BANDS, _standardize, make_splits

#: Small grid, searched on the validation stays for every split seed.
CONFIGS = (
    {"hidden": 64, "layers": 2, "dropout": 0.5, "weight_decay": 1e-3, "lr": 0.01},
    {"hidden": 64, "layers": 2, "dropout": 0.2, "weight_decay": 1e-2, "lr": 0.01},
)
MODELS = ("gcn", "gcn_w", "mlp", "gcn_nosame")
KEY = ["model", "graph", "source", "feature_set"]


def pick_device(device: str | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def has_weights(data) -> bool:
    """True if the saved network carries edge weights that are not all equal
    (otherwise `gcn_w` would be identical to `gcn`)."""
    w = getattr(data, "edge_weight", None)
    return w is not None and w.numel() > 0 and bool((w.max() - w.min()) > 1e-9)


def edges_for(data, model: str):
    """(edge_index, edge_weight or None): the links a given model may use. Both
    directions are already in `data.edge_index`; this only removes links."""
    ei = data.edge_index
    if model == "mlp":
        return ei[:, :0], None
    if model == "gcn_nosame":
        if getattr(data, "edge_attr", None) is None:
            raise ValueError("this Data has no same-patient flag (edge_attr)")
        same = data.edge_attr.reshape(ei.shape[1], -1)[:, 0] > 0.5
        return ei[:, ~same], None
    if model == "gcn":
        return ei, None
    if model == "gcn_w":
        if getattr(data, "edge_weight", None) is None:
            raise ValueError("this Data has no edge_weight")
        return ei, data.edge_weight.float()
    raise ValueError(f"model must be one of {MODELS}")


def build_net(in_channels: int, cfg: dict) -> GCN:
    return GCN(in_channels, cfg["hidden"], cfg["layers"], out_channels=1, dropout=cfg["dropout"])


def fit_one(X: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor, edge_weight, sp: dict, cfg: dict, seed: int,
            max_epochs: int = 200, patience: int = 25, min_epochs: int = 20) -> dict:
    """Train one GCN, stop on the best validation AUROC, return the scores of
    every stay at that epoch. `sp` holds boolean numpy masks."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = X.device
    tr = torch.as_tensor(sp["train"], device=dev)
    va = torch.as_tensor(sp["val"], device=dev)
    y_f = y.float()
    y_va = y[va].cpu().numpy()

    net = build_net(X.shape[1], cfg).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    hist = {"train_loss": [], "val_loss": [], "val_auc": []}      # one entry per epoch, for the training curves
    best_auc, best_epoch, best_scores, stale = -1.0, 0, None, 0
    for epoch in range(max_epochs):
        net.train()
        opt.zero_grad()
        logit = net(X, edge_index, edge_weight=edge_weight).view(-1)
        loss = F.binary_cross_entropy_with_logits(logit[tr], y_f[tr])
        loss.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            s = net(X, edge_index, edge_weight=edge_weight).view(-1)
        auc = roc_auc_score(y_va, s[va].cpu().numpy())
        hist["train_loss"].append(float(loss.detach()))                # measured with dropout ON, before the step
        hist["val_loss"].append(float(F.binary_cross_entropy_with_logits(s[va], y_f[va])))   # dropout OFF
        hist["val_auc"].append(float(auc))
        if auc > best_auc:
            best_auc, best_epoch, best_scores, stale = auc, epoch, s.cpu().numpy(), 0
        else:
            stale += 1
            if epoch >= min_epochs and stale >= patience:
                break
    return {"val_auc": best_auc, "epoch": best_epoch, "scores": best_scores, "history": hist}


def _prepare(data, model: str, seed: int, subset: np.ndarray | None, device: str | None):
    """Everything one run needs: tensors on the device, the splits, the links (and weights)."""
    dev = pick_device(device)
    X, y, groups = data.x.numpy(), data.y.numpy(), data.patient_idx.numpy()
    ei, ew = edges_for(data, model)
    if subset is not None:
        keep = np.flatnonzero(subset)
        remap = -np.ones(len(y), dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        e = ei.numpy()
        ok = subset[e[0]] & subset[e[1]]
        ei = torch.as_tensor(remap[e[:, ok]], dtype=torch.long)
        ew = None if ew is None else ew[torch.as_tensor(ok)]
        X, y, groups = X[keep], y[keep], groups[keep]
    sp = make_splits(y, groups, seed)                       # identical to the linear / GraphSAGE / GAT runs
    Xs = _standardize(X, data.x_scale_mask.numpy(), sp["train"])
    return {"X": torch.as_tensor(Xs, dtype=torch.float32, device=dev),
            "y_np": y, "y": torch.as_tensor(y, dtype=torch.long, device=dev),
            "groups": groups, "ei": ei.to(dev), "ew": None if ew is None else ew.to(dev),
            "sp": sp, "subset": subset}


def _select(prep: dict, configs, seed: int, **fit_kw) -> dict:
    best = None
    for cfg in configs:
        r = fit_one(prep["X"], prep["y"], prep["ei"], prep["ew"], prep["sp"], cfg, seed, **fit_kw)
        if best is None or r["val_auc"] > best["val_auc"]:
            best = {**r, "cfg": cfg}
    return best


def evaluate_data(data, model: str = "gcn", seed: int = 0, configs=CONFIGS, year: np.ndarray | None = None,
                  subset: np.ndarray | None = None, device: str | None = None, **fit_kw) -> dict:
    """One split seed on a PyG `Data` from `build_pyg_data` (reads x, y, edge_index,
    edge_weight, edge_attr, patient_idx, x_scale_mask). `subset`: boolean mask of the
    stays to use at all (e.g. those with a CLMBR vector); the network is restricted
    to links among them."""
    prep = _prepare(data, model, seed, subset, device)
    best = _select(prep, configs, seed, **fit_kw)
    y, ei, sp = prep["y_np"], prep["ei"], prep["sp"]
    if year is not None and subset is not None:
        year = year[subset]

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


def training_curves(data, model: str = "gcn", seed: int = 0, configs=CONFIGS, subset: np.ndarray | None = None,
                    device: str | None = None, **fit_kw) -> pd.DataFrame:
    """The per-epoch history of every setting in `configs` for one split seed, in long
    format: one row per (setting, epoch) with `train_loss`, `val_loss` and `val_auc`.

    It repeats exactly the runs `evaluate_data` does (same seed, same splits, same
    code), so the setting flagged `chosen` and its `best_epoch` are the ones behind
    that run's reported test AUROC. `train_loss` is measured with dropout ON (what
    the optimiser sees); `val_loss` and `val_auc` with dropout OFF. Epochs count from 0,
    as the `epoch` column of the results does. Nothing here touches the test stays."""
    prep = _prepare(data, model, seed, subset, device)
    best_i, best_auc, runs = None, -1.0, []
    for i, cfg in enumerate(configs):
        r = fit_one(prep["X"], prep["y"], prep["ei"], prep["ew"], prep["sp"], cfg, seed, **fit_kw)
        runs.append(r)
        if r["val_auc"] > best_auc:                                   # same tie rule as `_select`: first wins
            best_i, best_auc = i, r["val_auc"]
    rows = []
    for i, (cfg, r) in enumerate(zip(configs, runs)):
        h = r["history"]
        n = len(h["train_loss"])
        rows.append(pd.DataFrame({
            "seed": seed, "config": i, "dropout": cfg["dropout"], "weight_decay": cfg["weight_decay"],
            "chosen": i == best_i, "best_epoch": r["epoch"], "epoch": np.arange(n),
            "train_loss": h["train_loss"], "val_loss": h["val_loss"], "val_auc": h["val_auc"]}))
    return pd.concat(rows, ignore_index=True)


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
