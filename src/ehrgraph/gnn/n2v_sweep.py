"""Tuning node2vec for a one-hop embedding, without using the outcome labels.

Why not tune on the outcome? Choosing settings by how well the embedding predicts
long stay / high blood pressure (a) uses the labels to pick the settings, which
inflates whatever score is reported afterwards, and (b) if done on the honest
network only, hands that network a head start in the honest-vs-cheating
comparison. Here the criterion is instead

    link prediction: hide 10% of the links, train node2vec on the rest, and
    score how well cosine similarity separates the hidden links from random
    non-links.

With window=1 node2vec is trained to place *directly linked* stays close
together, so this measures exactly the objective. It needs no labels and can be
applied identically to every network.

Two details matter:
  * About a third of the links join two stays of the SAME patient (an easy
    target: the same person's other admission). The selection metric is
    therefore `auc_diff`, computed on hidden links between DIFFERENT patients.
  * Run-to-run noise (seed) is large relative to the differences between
    settings, so every setting is run with several seeds, summarised as mean
    and standard error, and chosen with a one-standard-error rule: the cheapest
    setting whose mean is within one standard error of the best.

The outcome AUROC of the finished embedding is a *check* (is it still useful?),
never a selection criterion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ehrgraph.gnn.node_features import _adjacency, node2vec_block

PARAM_COLS = ["window", "p", "q", "dim", "epochs", "negatives", "neg_power", "weighted", "num_walks", "walk_length"]


def split_edges(graph, nodes: list, frac: float = 0.1, seed: int = 0, tries: int = 50) -> dict:
    """Hide `frac` of the links. Returns the training graph, the hidden links
    (`pos`, as node positions), one random non-link per hidden link (`neg`;
    a different patient, not linked in the full graph, drawn with probability
    proportional to training degree so hubs are not trivially easy), and
    `pos_same` (True where a hidden link joins two stays of one patient)."""
    rng = np.random.default_rng(seed)
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(u, v) for u, v in graph.edges() if u != v]
    order = rng.permutation(len(edges))
    n_hold = int(frac * len(edges))
    held = [edges[i] for i in order[:n_hold]]
    train = graph.copy()
    train.remove_edges_from(held)

    deg = np.array([train.degree(n) if n in train else 0 for n in nodes], dtype=float)
    cand = np.flatnonzero(deg > 0)
    prob = deg[cand] / deg[cand].sum()
    pos = np.array([(idx[u], idx[v]) for u, v in held])
    pos_same = np.array([u[0] == v[0] for u, v in held])
    neg = np.zeros_like(pos)
    for k, (u, _) in enumerate(held):
        for _ in range(tries):
            w = int(rng.choice(cand, p=prob))
            nw = nodes[w]
            if nw != u and nw[0] != u[0] and not graph.has_edge(u, nw):
                break
        neg[k] = (idx[u], w)
    return {"train_graph": train, "pos": pos, "neg": neg, "pos_same": pos_same, "nodes": nodes}


def link_metrics(z: np.ndarray, ok: np.ndarray, split: dict) -> dict:
    """AUROC of cosine similarity: hidden links vs. random non-links.
    `ok`: nodes that have at least one training link (the others have zero
    vectors). auc_all uses every hidden link; auc_diff only those between
    different patients (the selection metric)."""
    zn = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-9, None)
    pos, neg = split["pos"], split["neg"]
    cp = (zn[pos[:, 0]] * zn[pos[:, 1]]).sum(1)
    cn = (zn[neg[:, 0]] * zn[neg[:, 1]]).sum(1)
    valid = ok[pos[:, 0]] & ok[pos[:, 1]] & ok[neg[:, 0]] & ok[neg[:, 1]]

    def auc(mask):
        m = valid & mask
        if m.sum() < 10:
            return float("nan")
        return float(roc_auc_score(np.r_[np.ones(m.sum()), np.zeros(m.sum())], np.r_[cp[m], cn[m]]))

    return {
        "auc_all": auc(np.ones(len(pos), dtype=bool)),
        "auc_diff": auc(~split["pos_same"]),
        "n_eval": int(valid.sum()),
    }


def heuristic_baseline(split: dict) -> dict:
    """Non-learned reference on the same hidden links: common neighbours and
    Jaccard overlap in the training graph. A learned embedding should be judged
    against these, not only against other settings of itself."""
    nodes = split["nodes"]
    adj, _ = _adjacency(split["train_graph"], nodes, False)
    nb = [set(a) for a in adj]
    ok = np.array([len(a) > 0 for a in nb])
    pos, neg = split["pos"], split["neg"]

    def score(pairs, kind):
        out = np.zeros(len(pairs))
        for k, (u, v) in enumerate(pairs):
            c = len(nb[u] & nb[v])
            out[k] = c if kind == "common_neighbors" else c / max(1, len(nb[u] | nb[v]))
        return out

    valid = ok[pos[:, 0]] & ok[pos[:, 1]] & ok[neg[:, 0]] & ok[neg[:, 1]]
    res = {}
    for kind in ("common_neighbors", "jaccard"):
        sp_, sn_ = score(pos, kind), score(neg, kind)
        for tag, mask in (("all", np.ones(len(pos), bool)), ("diff", ~split["pos_same"])):
            m = valid & mask
            res[f"{kind}_auc_{tag}"] = float(
                roc_auc_score(np.r_[np.ones(m.sum()), np.zeros(m.sum())], np.r_[sp_[m], sn_[m]])
            )
    return res


def label_of(cfg: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())) or "baseline"


def run_sweep(configs: list, targets: dict, base: dict, seeds=(0, 1, 2), frac=0.1, split_seed=0, log=print):
    """Train node2vec for every config x target x seed on the 90% training graph
    and score it on the hidden links.

    configs: list of dicts overriding `base` (keys of node2vec_block).
    targets: {name: (graph, nodes)} -- e.g. both networks of both questions, so
             the chosen setting is not tuned to a single network.
    The hidden links are fixed (split_seed) so settings are compared on the same
    links. Returns one row per (config, target, seed)."""
    splits = {name: split_edges(g, nodes, frac, split_seed) for name, (g, nodes) in targets.items()}
    rows = []
    for ci, cfg in enumerate(configs):
        full = {**base, **cfg}
        for name, (g, nodes) in targets.items():
            for seed in seeds:
                b = node2vec_block(splits[name]["train_graph"], nodes, **{**full, "seed": seed})
                m = link_metrics(b.values[:, :-1], b.values[:, -1] == 0, splits[name])
                rows.append({"config": label_of(cfg), "target": name, "seed": seed, **m,
                             **{k: full.get(k) for k in PARAM_COLS}})
        log(f"[{ci + 1}/{len(configs)}] {label_of(cfg)}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, metric: str = "auc_diff") -> pd.DataFrame:
    """Per config: mean of `metric` (averaged over targets within each seed),
    its standard error across seeds, the per-target means, and whether the mean
    is within one standard error of the best config."""
    per_seed = df.groupby(["config", "seed"])[metric].mean().reset_index()
    g = per_seed.groupby("config")[metric]
    out = pd.DataFrame({"score": g.mean(), "sd": g.std(), "n_seeds": g.count()})
    out["se"] = out["sd"] / np.sqrt(out["n_seeds"])
    out = out.join(df.groupby(["config", "target"])[metric].mean().unstack())
    params = df.drop_duplicates("config").set_index("config")[PARAM_COLS]
    out = out.join(params)
    best = out["score"].idxmax()
    se_best = 0.0 if pd.isna(out.loc[best, "se"]) else out.loc[best, "se"]   # one seed: no standard error
    out["within_1se"] = out["score"] >= out.loc[best, "score"] - se_best
    out["cost"] = out["dim"] * out["epochs"] * out["num_walks"] * out["walk_length"] * (out["negatives"] + 1)
    return out.sort_values("score", ascending=False)


def pick_one_se(summary: pd.DataFrame, prefer: str | None = None) -> str:
    """Choose a config by the one-standard-error rule: among configs whose mean
    is within one standard error of the best, take `prefer` (e.g. the neutral
    default) if it is among them, otherwise the cheapest (fewest dim x epochs x
    walks x negatives). Cheaper-or-default and just as good wins."""
    ok = summary[summary["within_1se"]]
    if prefer is not None and prefer in ok.index:
        return prefer
    return ok["cost"].idxmin()


def combine_winners(summary: pd.DataFrame, best: dict, base: dict, space: dict) -> dict:
    """Merge the one-factor-at-a-time results into one config. For each
    parameter in `space`, look at the rows that differ from the `best` config in
    that parameter only; adopt its best value only if it beats the `best`
    config's own score by more than that config's standard error, otherwise keep
    the current value. Settings change only when they are clearly better."""
    best_full = {**base, **best}
    ref = summary.loc[label_of(best)]
    margin = ref["score"] + (0.0 if pd.isna(ref["se"]) else ref["se"])
    final = dict(best)
    for k in space:
        mask = np.ones(len(summary), dtype=bool)
        for c in PARAM_COLS:
            if c != k:
                mask &= (summary[c] == best_full[c]).to_numpy()
        sub = summary[mask]
        row = sub["score"].idxmax()
        if sub.loc[row, "score"] > margin:
            v = sub.loc[row, k]
            final[k] = v.item() if hasattr(v, "item") else v
    return final


def ofat(best: dict, space: dict) -> list:
    """One-factor-at-a-time configs around `best`: the best itself, plus each
    value of each parameter in `space` with everything else held at `best`."""
    configs, seen = [dict(best)], {tuple(sorted(best.items()))}
    for k, vals in space.items():
        for v in vals:
            c = {**best, k: v}
            key = tuple(sorted(c.items()))
            if key not in seen:
                seen.add(key)
                configs.append(c)
    return configs
