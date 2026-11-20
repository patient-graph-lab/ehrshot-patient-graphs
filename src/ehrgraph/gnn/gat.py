"""Graph attention (GAT) on the patient-similarity networks, evaluated like the linear and GraphSAGE runs.

Same protocol as `linear_regressor.py` / `graph_sage.py`, so a GAT number can be put
next to a linear or GraphSAGE number directly:

  * the SAME patient-grouped, outcome-stratified 60 / 20 / 20 splits
    (`make_splits`, same split seeds);
  * columns standardised with the TRAIN stays' mean and std only;
  * the settings (`CONFIGS`) and the stopping epoch chosen on the VALIDATION
    stays; the TEST stays are scored once, at the end;
  * repeated over many split seeds, reported as mean and spread.

What GAT changes relative to GraphSAGE
--------------------------------------
GraphSAGE gives every linked stay the same weight (a mean). A graph attention
layer LEARNS a weight for each link: for stay i and neighbour j the layer scores
how useful j is for i, normalises the scores over i's neighbours (softmax), and
averages the neighbours with those weights. Several attention "heads" each learn
their own weights. So GAT can, in principle, listen to some neighbours (the same
patient's other stays, say) more than others, or put nearly all its weight on the
stay itself and ignore the network. `attention_report` shows which.

Details of `torch_geometric.nn.models.GAT` that matter here
  * `hidden_channels` is SPLIT across heads: hidden=64, heads=4 -> 16 per head.
  * Every stay also attends to ITSELF (`add_self_loops=True`). With every link
    removed each stay attends only to itself, which is an ordinary MLP. That is
    the "mlp" control below, exact.
  * Like GraphSAGE it does NOT use the stored edge weights
    (`supports_edge_weight` is False). It can take edge features (`edge_dim`);
    this module does not give it any, so the links enter as present / absent.
  * With dropout p, p is applied both to the hidden features and to the
    attention coefficients themselves.
  * Full batch and transductive, exactly as in `graph_sage.py`: the whole network
    is visible, only TRAIN stays' outcomes enter the loss, and a patient's stays
    all sit in one split.

Controls (the `model` column of the results)
  * "gat"         : GAT on the saved network.
  * "mlp"         : the SAME architecture and training with every link removed.
                    "gat minus mlp" on identical splits is what the network adds.
  * "gat_nosame"  : GAT with the links between two stays of the SAME patient
                    removed (about a third of the links).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn.models import GAT

from ehrgraph.gnn.linear_regressor import YEAR_BANDS, _standardize, make_splits

#: Small grid, searched on the validation stays for every split seed.
#: hidden is split across `heads` (64 / 4 = 16 per head). `v2=True` would use GATv2.
CONFIGS = (
    {"hidden": 64, "heads": 4, "layers": 2, "dropout": 0.5, "weight_decay": 1e-3, "lr": 0.01},
    {"hidden": 64, "heads": 4, "layers": 2, "dropout": 0.2, "weight_decay": 1e-2, "lr": 0.01},
)
MODELS = ("gat", "mlp", "gat_nosame")
KEY = ["model", "graph", "source", "feature_set"]


def pick_device(device: str | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def edges_for(data, model: str) -> torch.Tensor:
    """The links a given model may use. Both directions are already in
    `data.edge_index`; this only removes links."""
    ei = data.edge_index
    if model == "mlp":
        return ei[:, :0]
    if model == "gat_nosame":
        if getattr(data, "edge_attr", None) is None:
            raise ValueError("this Data has no same-patient flag (edge_attr)")
        same = data.edge_attr.reshape(ei.shape[1], -1)[:, 0] > 0.5
        return ei[:, ~same]
    if model == "gat":
        return ei
    raise ValueError(f"model must be one of {MODELS}")


def build_net(in_channels: int, cfg: dict) -> GAT:
    return GAT(in_channels, cfg["hidden"], cfg["layers"], out_channels=1, dropout=cfg["dropout"],
               heads=cfg["heads"], v2=cfg.get("v2", False))


def fit_one(X: torch.Tensor, y: torch.Tensor, edge_index: torch.Tensor, sp: dict, cfg: dict, seed: int,
            max_epochs: int = 200, patience: int = 25, min_epochs: int = 20, return_model: bool = False) -> dict:
    """Train one GAT, stop on the best validation AUROC, return the scores of
    every stay at that epoch (and, if asked, the network at that epoch).
    `sp` holds boolean numpy masks."""
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
    best_auc, best_epoch, best_scores, best_state, stale = -1.0, 0, None, None, 0
    for epoch in range(max_epochs):
        net.train()
        opt.zero_grad()
        logit = net(X, edge_index).view(-1)
        loss = F.binary_cross_entropy_with_logits(logit[tr], y_f[tr])
        loss.backward()
        opt.step()

        net.eval()
        with torch.no_grad():
            s = net(X, edge_index).view(-1)
        auc = roc_auc_score(y_va, s[va].cpu().numpy())
        hist["train_loss"].append(float(loss.detach()))                # measured with dropout ON, before the step
        hist["val_loss"].append(float(F.binary_cross_entropy_with_logits(s[va], y_f[va])))   # dropout OFF
        hist["val_auc"].append(float(auc))
        if auc > best_auc:
            best_auc, best_epoch, best_scores, stale = auc, epoch, s.cpu().numpy(), 0
            if return_model:
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if epoch >= min_epochs and stale >= patience:
                break
    out = {"val_auc": best_auc, "epoch": best_epoch, "scores": best_scores, "history": hist}
    if return_model:
        net.load_state_dict(best_state)
        out["net"] = net.eval()
    return out


def _prepare(data, model: str, seed: int, subset: np.ndarray | None, device: str | None):
    """Everything one run needs: tensors on the device, the splits, the links."""
    dev = pick_device(device)
    X, y, groups = data.x.numpy(), data.y.numpy(), data.patient_idx.numpy()
    ei = edges_for(data, model)
    if subset is not None:
        keep = np.flatnonzero(subset)
        remap = -np.ones(len(y), dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        e = ei.numpy()
        ok = subset[e[0]] & subset[e[1]]
        ei = torch.as_tensor(remap[e[:, ok]], dtype=torch.long)
        X, y, groups = X[keep], y[keep], groups[keep]
    sp = make_splits(y, groups, seed)                       # identical to the linear / GraphSAGE runs
    Xs = _standardize(X, data.x_scale_mask.numpy(), sp["train"])
    return {"X": torch.as_tensor(Xs, dtype=torch.float32, device=dev),
            "y_np": y, "y": torch.as_tensor(y, dtype=torch.long, device=dev),
            "groups": groups, "ei": ei.to(dev), "sp": sp, "subset": subset}


def _select(prep: dict, configs, seed: int, **fit_kw) -> dict:
    best = None
    for cfg in configs:
        r = fit_one(prep["X"], prep["y"], prep["ei"], prep["sp"], cfg, seed, **fit_kw)
        if best is None or r["val_auc"] > best["val_auc"]:
            best = {**r, "cfg": cfg}
    return best


def evaluate_data(data, model: str = "gat", seed: int = 0, configs=CONFIGS, year: np.ndarray | None = None,
                  subset: np.ndarray | None = None, device: str | None = None, **fit_kw) -> dict:
    """One split seed on a PyG `Data` from `build_pyg_data` (reads x, y, edge_index,
    edge_attr, patient_idx, x_scale_mask). `subset`: boolean mask of the stays to
    use at all (e.g. those with a CLMBR vector); the network is restricted to
    links among them."""
    prep = _prepare(data, model, seed, subset, device)
    best = _select(prep, configs, seed, **fit_kw)
    y, te, s = prep["y_np"], prep["sp"]["test"], best["scores"]
    if year is not None and subset is not None:
        year = year[subset]
    out = {"seed": seed, "n_used": int(len(y)), "n_edges": int(prep["ei"].shape[1]),
           "hidden": best["cfg"]["hidden"], "heads": best["cfg"]["heads"], "dropout": best["cfg"]["dropout"],
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


def training_curves(data, model: str = "gat", seed: int = 0, configs=CONFIGS, subset: np.ndarray | None = None,
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
        r = fit_one(prep["X"], prep["y"], prep["ei"], prep["sp"], cfg, seed, **fit_kw)
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


# ---------------------------------------------------------------- attention
def _capture_attention(net: GAT, X: torch.Tensor, edge_index: torch.Tensor) -> list:
    """Run the trained network once (eval mode) and return, per layer,
    (edge_index including the added self-loops, attention [edges, heads]).
    Hooks ask each GATConv for its attention without changing what it computes."""
    caps = []

    def pre(mod, args, kwargs):
        kwargs = dict(kwargs)
        kwargs["return_attention_weights"] = True
        return args, kwargs

    def post(mod, args, kwargs, out):
        caps.append(out[1])
        return out[0]

    handles = []
    for conv in net.convs:
        handles += [conv.register_forward_pre_hook(pre, with_kwargs=True),
                    conv.register_forward_hook(post, with_kwargs=True)]
    try:
        net.eval()
        with torch.no_grad():
            net(X, edge_index)
    finally:
        for h in handles:
            h.remove()
    return caps


def attention_report(data, model: str = "gat", seed: int = 0, configs=CONFIGS, subset: np.ndarray | None = None,
                     device: str | None = None, **fit_kw) -> pd.DataFrame:
    """Train one GAT exactly as `evaluate_data` does (same split seed, same
    settings search) and describe where its attention goes, one row per layer,
    averaged over heads and over stays that have at least one link:

      self_share            weight a stay puts on ITSELF
      uniform_self_share    what it would be if all links and the stay itself were weighted equally
      entropy_ratio         1 = perfectly even weights over links, 0 = all weight on one
      same_patient_att      weight on the SAME patient's other stays (stays that have both kinds of link)
      same_patient_links    the share of links that are same-patient, for those stays

    Read it as: attention near `uniform_*` and entropy near 1 means the layer is not
    selecting among neighbours; `self_share` far above uniform means it mostly ignores
    the network; same_patient_att far from same_patient_links means it prefers (or avoids)
    the same person's other admissions.

    A limit: a link's attention score is computed from the features of its two stays
    only (no link feature is given), so attention can favour a kind of link only if
    that kind is recognisable from the stays' features. Uniform attention therefore
    means "no selection that these features allow", not "all links are equally good"."""
    prep = _prepare(data, model, seed, subset, device)
    best = _select(prep, configs, seed, return_model=True, **fit_kw)
    caps = _capture_attention(best["net"], prep["X"], prep["ei"])
    df = summarize_attention(caps, prep["groups"], prep["X"].shape[0])
    df.insert(1, "model", model)
    df.insert(2, "split_seed", seed)
    df.insert(3, "config", [best["cfg"]] * len(df))
    return df


def summarize_attention(caps: list, groups: np.ndarray, N: int) -> pd.DataFrame:
    """The per-layer numbers of `attention_report`, from captured attention
    (`caps`, as `_capture_attention` returns) and the patient of every stay."""
    pat = torch.as_tensor(groups, device=caps[0][0].device)
    rows = []
    for layer, (ei, alpha) in enumerate(caps):
        a = alpha.mean(1)
        src, dst = ei[0], ei[1]
        is_self = src == dst
        nb = ~is_self
        zeros = lambda: torch.zeros(N, device=a.device)
        deg = zeros().index_add_(0, dst[nb], torch.ones(int(nb.sum()), device=a.device))
        has = deg > 0
        self_w = zeros().index_add_(0, dst[is_self], a[is_self])
        ent = zeros().index_add_(0, dst, -(a * torch.log(a.clamp_min(1e-12))))
        same = nb & (pat[src] == pat[dst])
        n_same = zeros().index_add_(0, dst[same], torch.ones(int(same.sum()), device=a.device))
        w_same = zeros().index_add_(0, dst[same], a[same])
        w_nb = zeros().index_add_(0, dst[nb], a[nb])
        both = has & (n_same > 0) & (n_same < deg)
        rows.append({
            "layer": layer + 1,
            "n_stays_with_links": int(has.sum()),
            "self_share": float(self_w[has].mean()),
            "uniform_self_share": float((1.0 / (deg[has] + 1)).mean()),
            "entropy_ratio": float((ent[has] / torch.log(deg[has] + 1)).mean()),
            "n_stays_both_kinds": int(both.sum()),
            "same_patient_att": float((w_same[both] / w_nb[both]).mean()) if both.any() else float("nan"),
            "same_patient_links": float((n_same[both] / deg[both]).mean()) if both.any() else float("nan"),
        })
    return pd.DataFrame(rows)
