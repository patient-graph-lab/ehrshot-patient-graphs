"""Compare the linear baseline and the three graph networks (GraphSAGE, GAT, GCN).

Nothing is retrained here. The four earlier notebooks each saved a table with one
row per (model, network, diagnosis source, feature set, split seed):

    13  linear_baselines.csv     linear / logistic model, no links
    14  graphsage_results.csv    sage, mlp, sage_nosame
    15  gat_results.csv          gat,  mlp, gat_nosame
    16  gcn_results.csv          gcn,  gcn_w, mlp, gcn_nosame

All four used the same patient-grouped splits for the same split seed, so a
difference between two models is taken PAIRED by seed (split luck cancels), over
the seeds both models ran.

Naming
------
The "mlp" rows of notebooks 14, 15 and 16 are the same idea (the network with every
link removed) trained through each architecture's own code, so they are labelled
"MLP via GraphSAGE" / "MLP via GAT" / "MLP via GCN". They should agree; how much
they differ is a noise floor for the differences between architectures.

A guard that matters
--------------------
A comparison is returned only if both models were scored on the SAME stays
(identical `n_used` for every shared seed). Runs restricted to a subset (e.g. stays
with a CLMBR vector) have different splits from runs on all stays, and pairing them
would be meaningless, so they are skipped, never silently mixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: (file, {model name in that file: (architecture, kind)})
SOURCES = {
    "linear": ("linear_baselines.csv", None),
    "GraphSAGE": ("graphsage_results.csv", {"sage": "network", "mlp": "no links", "sage_nosame": "no same-patient links"}),
    "GAT": ("gat_results.csv", {"gat": "network", "mlp": "no links", "gat_nosame": "no same-patient links"}),
    "GCN": ("gcn_results.csv", {"gcn": "network", "gcn_w": "network, weighted", "mlp": "no links",
                                "gcn_nosame": "no same-patient links"}),
}
#: Models that take part in ranks, the pairwise matrix and the headline table.
CORE = ["Linear", "MLP via GraphSAGE", "MLP via GAT", "MLP via GCN", "GraphSAGE", "GAT", "GCN"]
ARCHS = ["GraphSAGE", "GAT", "GCN"]
HONEST = ("safe", "safe", "dx")          # honest network, honest diagnoses, diagnosis features


def label_of(arch: str, kind: str) -> str:
    return {"network": arch, "network, weighted": f"{arch} (weighted)", "no links": f"MLP via {arch}",
            "no same-patient links": f"{arch} (no same-patient links)"}[kind]


def load_all(results_dir) -> tuple[pd.DataFrame, list]:
    """One long table of every available result, with `label`, `arch` and `kind`
    columns added, and the list of files that were not found."""
    frames, missing = [], []
    for arch, (fname, mapping) in SOURCES.items():
        f = results_dir / fname
        if not f.exists():
            missing.append(fname)
            continue
        d = pd.read_csv(f)
        if mapping is None:
            d["arch"], d["kind"], d["label"] = "Linear", "no links", "Linear"
        else:
            unknown = set(d["model"]) - set(mapping)
            if unknown:
                raise ValueError(f"{fname}: unrecognised model names {sorted(unknown)}")
            d["arch"] = arch
            d["kind"] = d["model"].map(mapping)
            d["label"] = [label_of(arch, k) for k in d["kind"]]
        if "n_used" not in d:
            d["n_used"] = np.nan
        frames.append(d)
    if not frames:
        raise FileNotFoundError("none of the result files were found; run notebooks 13 to 16 first")
    return pd.concat(frames, ignore_index=True), missing


def get(D: pd.DataFrame, q: str, label: str, graph: str, source: str, fs: str) -> pd.DataFrame:
    """The rows of one model in one scenario, indexed by split seed."""
    m = ((D.question == q) & (D.label == label) & (D.graph == graph) & (D.source == source) & (D.feature_set == fs))
    s = D[m].set_index("seed")
    if s.index.duplicated().any():
        raise ValueError(f"duplicate rows for {q} | {label} | {graph}/{source}/{fs}: results were appended twice?")
    return s


def pair(D: pd.DataFrame, q: str, a: tuple, b: tuple, metric: str = "test_auc") -> dict | None:
    """a minus b, paired by split seed. a, b = (label, graph, source, feature_set).
    None if there are no shared seeds, or the two were scored on different stays."""
    A, B = get(D, q, *a), get(D, q, *b)
    seeds = A.index.intersection(B.index)
    if len(seeds) == 0:
        return None
    if (A.loc[seeds, "n_used"].to_numpy() != B.loc[seeds, "n_used"].to_numpy()).any():
        return None
    d = A.loc[seeds, metric] - B.loc[seeds, metric]
    return {"mean_diff": float(d.mean()), "sd": float(d.std()), "n_seeds": int(len(d)),
            "fraction_a_higher": float((d > 0).mean())}


def pm(r: dict | None, digits: int = 3) -> str:
    """'mean ± sd' for a `pair` result (blank if there is none)."""
    if r is None:
        return ""
    sd = "nan" if np.isnan(r["sd"]) else f"{r['sd']:.{digits}f}"
    return f"{r['mean_diff']:+.{digits}f} ± {sd}"


def year_alone(y: np.ndarray, groups: np.ndarray, years: np.ndarray, seeds, model: str = "logistic") -> pd.DataFrame:
    """The prediction year as the only feature, on the same splits (same code as
    notebook 13): the reference every model has to clear."""
    from ehrgraph.gnn.linear_regressor import evaluate_arrays

    yr = np.asarray(years, dtype=float)[:, None]
    return pd.DataFrame([evaluate_arrays(yr, np.array([True]), y, groups, s, model) for s in seeds])


def _arch_of(D: pd.DataFrame, label: str) -> str:
    return D.loc[D.label == label, "arch"].iloc[0]


def common_seeds(D: pd.DataFrame, q: str, labels: list, scenario: tuple = HONEST) -> pd.Index:
    """The split seeds every one of `labels` ran in this scenario. Means over
    different seed sets are not comparable (the linear notebook ran more seeds than
    the network notebooks), so the tables below use these seeds only."""
    idx = None
    for l in labels:
        s = get(D, q, l, *scenario).index
        idx = s if idx is None else idx.intersection(s)
    return idx if idx is not None else pd.Index([])


def leaderboard(D: pd.DataFrame, q: str, scenario: tuple = HONEST) -> pd.DataFrame:
    """Every model that has this scenario: mean / spread of test AUROC, mean rank and
    share of splits won (core models only), and the paired difference to the linear
    model and to the model's own no-links MLP. All numbers use the seeds that EVERY
    listed model ran (`n_seeds` says how many), so the rows are comparable."""
    graph, source, fs = scenario
    labels = [l for l in CORE + [x for x in D.label.unique() if x not in CORE]
              if l in set(D.label) and len(get(D, q, l, graph, source, fs))]
    seeds = common_seeds(D, q, labels, scenario)
    if len(seeds) == 0:
        raise ValueError(f"{q}: no split seed is shared by all models; rerun the notebooks with the same SEEDS")
    core = [l for l in CORE if l in labels]
    ranks = None
    if len(core) >= 2:
        wide = pd.concat({l: get(D, q, l, graph, source, fs).loc[seeds, "test_auc"] for l in core}, axis=1).dropna()
        if len(wide):
            ranks = wide.rank(axis=1, ascending=False)
    rows = []
    for l in labels:
        g = get(D, q, l, graph, source, fs).loc[seeds]
        arch = _arch_of(D, l)
        own_mlp = "Linear" if arch == "Linear" else f"MLP via {arch}"
        vs_lin = pair(D, q, (l, *scenario), ("Linear", *scenario)) if l != "Linear" else None
        vs_mlp = pair(D, q, (l, *scenario), (own_mlp, graph if arch == "Linear" else "safe", source, fs)) \
            if l not in (own_mlp, "Linear") else None
        rows.append({"model": l, "kind": D.loc[D.label == l, "kind"].iloc[0], "n_seeds": len(g),
                     "mean": g["test_auc"].mean(), "sd": g["test_auc"].std(), "test_ap": g["test_ap"].mean(),
                     "mean_rank": ranks[l].mean() if ranks is not None and l in ranks else np.nan,
                     "share_of_splits_won": float((ranks[l] == 1).mean()) if ranks is not None and l in ranks else np.nan,
                     "vs_linear": pm(vs_lin), "vs_linear_frac": vs_lin["fraction_a_higher"] if vs_lin else np.nan,
                     "vs_own_mlp": pm(vs_mlp), "vs_own_mlp_frac": vs_mlp["fraction_a_higher"] if vs_mlp else np.nan})
    return pd.DataFrame(rows).set_index("model")


def network_effects(D: pd.DataFrame, q: str) -> pd.DataFrame:
    """What the network adds, per architecture (all differences are A minus B, paired)."""
    rows = []
    H = HONEST
    for arch in ARCHS:
        net, mlp, nosame = arch, f"MLP via {arch}", f"{arch} (no same-patient links)"
        tests = [
            ("network adds (honest net):  network - MLP", (net, *H), (mlp, *H)),
            ("network adds (cheating net):  network - MLP", (net, "leaky", "safe", "dx"), (mlp, *H)),
            ("network adds (random net):  network - MLP", (net, "null", "safe", "dx"), (mlp, *H)),
            ("real vs random network", (net, *H), (net, "null", "safe", "dx")),
            ("cheating network vs honest", (net, "leaky", "safe", "dx"), (net, *H)),
            ("without same-patient links:  no-same - MLP", (nosame, *H), (mlp, *H)),
            ("same-patient links matter?  network - no-same", (net, *H), (nosame, *H)),
        ]
        if arch == "GCN":
            tests += [("edge weights add (honest net):  weighted - unweighted", ("GCN (weighted)", *H), ("GCN", *H)),
                      ("edge weights add (cheating net):  weighted - unweighted",
                       ("GCN (weighted)", "leaky", "safe", "dx"), ("GCN", "leaky", "safe", "dx"))]
        for name, a, b in tests:
            r = pair(D, q, a, b)
            if r is not None:
                rows.append({"architecture": arch, "comparison": name, **r})
    return pd.DataFrame(rows)


def cheating_effects(D: pd.DataFrame, q: str) -> pd.DataFrame:
    """Cheating diagnoses (the whole record, future included) vs honest ones, for every
    model, with the honest network; and both-cheat vs both-honest for the networks."""
    rows = []
    for l in CORE:
        if l not in set(D.label):
            continue
        r = pair(D, q, (l, "safe", "leaky", "dx"), (l, "safe", "safe", "dx"))
        if r is not None:
            rows.append({"model": l, "comparison": "cheating diagnoses vs honest (honest net)", **r})
        if l in ARCHS:
            r = pair(D, q, (l, "leaky", "leaky", "dx"), (l, "safe", "safe", "dx"))
            if r is not None:
                rows.append({"model": l, "comparison": "both cheat vs both honest", **r})
    return pd.DataFrame(rows)


def pairwise(D: pd.DataFrame, q: str, labels=None, scenario: tuple = HONEST) -> dict:
    """Row model minus column model, paired by seed: mean difference, spread, and the
    share of splits where the row model won."""
    labels = [l for l in (labels or CORE) if l in set(D.label) and len(get(D, q, l, *scenario))]
    mean = pd.DataFrame(np.nan, index=labels, columns=labels)
    sd, frac = mean.copy(), mean.copy()
    for a in labels:
        for b in labels:
            if a == b:
                continue
            r = pair(D, q, (a, *scenario), (b, *scenario))
            if r is not None:
                mean.loc[a, b], sd.loc[a, b], frac.loc[a, b] = r["mean_diff"], r["sd"], r["fraction_a_higher"]
    return {"mean": mean, "sd": sd, "frac": frac}
