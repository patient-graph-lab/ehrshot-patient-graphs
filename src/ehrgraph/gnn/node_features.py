"""Node features and input embeddings for the GNN experiments, as a library call.

Goal (leaky vs. non-leaky comparison): give GCN / GraphSAGE / GAT, and the
CLMBR baseline, node inputs built the *same way* for every graph, so any
difference between the honest and the cheating condition comes from the
information in the data, not from how the features were made.

Nothing here uses outcomes to build a feature. Labels appear only in the
importance diagnostics at the bottom. Nothing here makes a train/test split:
`build_pyg_data` returns unscaled features plus a `patient_idx` vector so that
patient-grouped splits and train-only standardisation can be added later
(`standardize`).

Feature blocks, in the vocabulary of the node-embeddings lecture:

  dx_svd, dx_summary   features of the node itself (diagnosis embedding)   CAN LEAK
  structure, rwpe      hand-engineered graph features ("traditional ML")   leaks via the graph
  n2v                  node2vec: LEARNED shallow embedding of the graph    leaks via the graph
  clmbr                Stanford's pretrained patient embedding             honest by construction
  time                 prediction year (ablation only)

The 2x2 design that separates *graph* leakage from *feature* leakage:

                      features: safe        features: leaky
      graph: safe     honest                leaky features only
      graph: leaky    leaky edges only      fully leaky
      graph: null     control: features with meaningless edges

`tg` is an `ehrgraph` TaskGraphs; only its attributes are used (nodes, labels,
graphs, vectors, features, feature_nodes, task), so tests can pass a stand-in.
torch / torch_geometric / gensim are imported lazily.
"""

from __future__ import annotations

import bisect
import itertools
import random
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

#: Which blocks go into a model's node features. CLMBR is kept out of the
#: diagnosis sets: it is identical under every condition, so mixing it in would
#: dilute exactly the honest-vs-cheating difference we want to measure.
FEATURE_SETS = {
    "dx": ["dx_svd", "dx_summary"],
    "clmbr": ["clmbr"],
    "structure": ["structure", "rwpe"],
    "n2v": ["n2v"],
    "dx_n2v": ["dx_svd", "dx_summary", "n2v"],
    "clmbr_n2v": ["clmbr", "n2v"],
}
#: Blocks that depend on the diagnosis vectors, i.e. that can leak through `source`.
LEAKABLE_BLOCKS = ("dx_svd", "dx_summary")


@dataclass
class FeatureBlock:
    name: str
    values: np.ndarray  # (n_nodes, d) float32, rows in tg.nodes order
    columns: list
    scale: bool = True  # standardise (train nodes only) when `standardize` is called
    meta: dict = field(default_factory=dict)


# ------------------------------------------------- features of the node itself


def tfidf_svd_block(matrix, codes: list, dim: int = 64, seed: int = 0, name: str = "dx_svd"):
    """Diagnosis embedding: TF-IDF of the node's diagnosis vector, then a
    truncated SVD (LSA). Same weighting idea as the similarity graph, so the
    embedding and the edges describe the same notion of 'similar diagnoses'.
    Unsupervised. In the lecture's terms: a deep-ish *feature* encoder, linear
    here, with no graph in it. `top_codes` shows which codes load on an axis."""
    tfidf = TfidfTransformer().fit_transform(sp.csr_matrix(matrix))
    dim = min(dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    z = svd.fit_transform(tfidf)
    return FeatureBlock(
        name,
        z.astype(np.float32),
        [f"{name}_{i}" for i in range(dim)],
        meta={
            "components": svd.components_,
            "codes": list(codes),
            "explained_variance": float(svd.explained_variance_ratio_.sum()),
        },
    )


def top_codes(block: FeatureBlock, component: int, n: int = 10) -> pd.DataFrame:
    """The diagnosis codes with the largest loadings on one SVD component."""
    load = block.meta["components"][component]
    order = np.argsort(-np.abs(load))[:n]
    return pd.DataFrame({"code": [block.meta["codes"][i] for i in order], "loading": load[order]})


def diagnosis_summary_block(matrix, name: str = "dx_summary") -> FeatureBlock:
    """How many distinct diagnoses the node has: the simplest leak carrier
    (a cheating profile has about twice as many, 97 vs 48 for long stay)."""
    n_codes = np.asarray((sp.csr_matrix(matrix) > 0).sum(axis=1)).ravel()
    return FeatureBlock(name, np.log1p(n_codes)[:, None].astype(np.float32), ["log_n_diagnoses"])


def time_block(nodes: list, name: str = "time") -> FeatureBlock:
    """Prediction year. In no feature set: with the 2015 ICD-9 -> ICD-10 switch,
    year is an easy shortcut. Use it as an ablation."""
    year = np.array([pd.Timestamp(n[1]).year for n in nodes], dtype=np.float32)
    return FeatureBlock(name, year[:, None], ["prediction_year"])


def clmbr_block(tg, dim: int | None = None, seed: int = 0) -> FeatureBlock | None:
    """Stanford's CLMBR patient embedding at the prediction time, aligned to
    tg.nodes. Nodes without a fingerprint get zeros plus `has_clmbr`=0 (the
    graphs keep them). None when the full release isn't loaded (tg.features is
    None). `dim`: optionally PCA down, e.g. to match the diagnosis embedding.
    tg.features is already centred over the fingerprinted nodes."""
    if tg.features is None:
        return None
    emb = np.asarray(tg.features, dtype=np.float64)
    if dim is not None:
        emb = PCA(n_components=dim, random_state=seed).fit_transform(emb)
    row = {n: i for i, n in enumerate(tg.feature_nodes)}
    out = np.zeros((len(tg.nodes), emb.shape[1] + 1), dtype=np.float32)
    for i, n in enumerate(tg.nodes):
        j = row.get(n)
        if j is not None:
            out[i, :-1] = emb[j]
            out[i, -1] = 1.0
    cols = [f"clmbr_{i}" for i in range(emb.shape[1])] + ["has_clmbr"]
    return FeatureBlock("clmbr", out, cols, scale=False)


# ------------------------------------------------- hand-engineered graph features


def structure_block(graph: nx.Graph, nodes: list, centrality: pd.DataFrame | None = None):
    """Label-free graph statistics per node (the lecture's 'traditional ML'
    feature engineering). `centrality`: the DataFrame from
    `analyze_task(...).centrality[condition]` (optional; adds betweenness,
    closeness, eigenvector as within-graph percentile ranks, which tame the
    heavy skew). Undefined centralities become 0."""
    g = nx.Graph(graph)
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    deg, clust, core = dict(g.degree()), nx.clustering(g), nx.core_number(g)
    cols = {
        "log_degree": [np.log1p(deg[n]) for n in nodes],
        "clustering": [clust[n] for n in nodes],
        "core_number": [core[n] for n in nodes],
        "n_same_patient_neighbors": [sum(m[0] == n[0] for m in g[n]) for n in nodes],
    }
    if centrality is not None:
        for m in ("betweenness", "closeness", "eigenvector"):
            if m in centrality:
                d = centrality[m].to_dict()
                raw = pd.Series([d.get(n, np.nan) for n in nodes])
                cols[f"{m}_pct"] = raw.rank(pct=True).fillna(0.0).to_numpy()
    values = np.column_stack([np.asarray(v, dtype=np.float64) for v in cols.values()])
    return FeatureBlock("structure", values.astype(np.float32), list(cols))


def random_walk_pe_block(graph: nx.Graph, nodes: list, steps: int = 8) -> FeatureBlock:
    """Random-walk *return probabilities*: the chance a t-step walk from a node
    is back at it, t = 1..steps. Computed exactly from matrix powers (nothing is
    learned, nothing is sampled). It describes a node's structural ROLE (triangle
    members score high), which node2vec cannot do (lecture, limitation 2), so it
    complements `n2v` rather than competing with it. Isolated nodes get 0."""
    a = sp.csr_matrix(nx.to_scipy_sparse_array(graph, nodelist=nodes, weight=None, format="csr"))
    deg = np.asarray(a.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=float), where=deg > 0)
    p = sp.csr_matrix(sp.diags(inv) @ a)
    m, out = p.copy(), []
    for t in range(steps):
        out.append(m.diagonal())
        if t < steps - 1:
            m = m @ p
    return FeatureBlock(
        "rwpe", np.column_stack(out).astype(np.float32), [f"return_prob_t{t + 1}" for t in range(steps)]
    )


# ------------------------------------------------- node2vec (learned embedding)


def _adjacency(graph: nx.Graph, nodes: list, weighted: bool):
    idx = {n: i for i, n in enumerate(nodes)}
    adj = [[] for _ in nodes]
    wts = [[] for _ in nodes] if weighted else None
    for u, v, d in graph.edges(data=True):
        if u == v:
            continue
        i, j = idx[u], idx[v]
        adj[i].append(j)
        adj[j].append(i)
        if weighted:
            w = float(d.get("weight", 1.0))
            wts[i].append(w)
            wts[j].append(w)
    return adj, wts


def node2vec_walks(adj, wts=None, num_walks=10, walk_length=30, p=1.0, q=1.0, seed=0):
    """Biased 2nd-order random walks (node2vec, Grover & Leskovec).

    A walker that just crossed edge (t, w) picks the next node x among w's
    neighbours with unnormalised weight  1/p if x == t  (distance 0 from t),
    1 if x is also a neighbour of t (distance 1), 1/q otherwise (distance 2),
    times the edge weight if `wts` is given. p = q = 1 is DeepWalk. Walks start
    from every node with at least one edge, `num_walks` times, in shuffled order;
    isolated nodes cannot walk and are skipped. Pure python, deterministic given
    `seed`; transition tables are cached per directed edge."""
    rng = random.Random(seed)
    nset = [set(a) for a in adj]
    cache: dict = {}

    def first(cur):
        ws = wts[cur] if wts else [1.0] * len(adj[cur])
        cum = list(itertools.accumulate(ws))
        return adj[cur][bisect.bisect_left(cum, rng.random() * cum[-1])]

    def step(prev, cur):
        cum = cache.get((prev, cur))
        if cum is None:
            ws = []
            for k, x in enumerate(adj[cur]):
                a = 1.0 / p if x == prev else (1.0 if x in nset[prev] else 1.0 / q)
                ws.append(a * (wts[cur][k] if wts else 1.0))
            cum = cache[(prev, cur)] = list(itertools.accumulate(ws))
        return adj[cur][bisect.bisect_left(cum, rng.random() * cum[-1])]

    starts = [v for v in range(len(adj)) if adj[v]]
    walks = []
    for _ in range(num_walks):
        rng.shuffle(starts)
        for s in starts:
            walk = [s, first(s)]
            while len(walk) < walk_length:
                walk.append(step(walk[-2], walk[-1]))
            walks.append(walk)
    return walks


def node2vec_block(
    graph: nx.Graph,
    nodes: list,
    dim: int = 64,
    num_walks: int = 10,
    walk_length: int = 30,
    window: int = 5,
    negatives: int = 5,
    epochs: int = 5,
    p: float = 1.0,
    q: float = 1.0,
    weighted: bool = False,
    neg_power: float = 1.0,
    seed: int = 0,
    name: str = "n2v",
) -> FeatureBlock:
    """node2vec embedding of one graph: biased walks -> skip-gram with negative
    sampling (gensim Word2Vec, sg=1), exactly the lecture's pipeline.

      walk_length, num_walks   short fixed-length walks from every node (r, l)
      window                   context size: N_R(u) = nodes within `window` steps
                               along the walks (the multiset of the slides)
      negatives                k noise nodes per observed pair (lecture: 5-20)
      neg_power                noise ~ (walk frequency)^neg_power; walk frequency
                               is roughly proportional to degree, so 1.0 is the
                               lecture's 'proportional to degree'; word2vec's
                               default is 0.75
      p, q                     return / in-out parameters
      weighted                 multiply transition weights by the similarity
                               weight (the lecture assumes a binary A)

    Embeddings are a free lookup table Z, one vector per node, learned with SGD
    (single worker, fixed seed, so reproducible). Three consequences:
    isolated nodes cannot take part in any walk, so they get a zero vector and a
    `n2v_isolated` flag (a random initial vector would be noise posing as
    signal); embeddings are only comparable WITHIN one graph (rotation is
    arbitrary), never between the safe and the leaky graph; and different seeds
    give different embeddings."""
    from gensim.models import Word2Vec

    adj, wts = _adjacency(graph, nodes, weighted)
    walks = node2vec_walks(adj, wts, num_walks, walk_length, p, q, seed)
    model = Word2Vec(
        sentences=[[str(v) for v in w] for w in walks],
        vector_size=dim,
        window=window,
        min_count=1,
        sg=1,
        hs=0,
        negative=negatives,
        ns_exponent=neg_power,
        sample=0,
        workers=1,
        epochs=epochs,
        seed=seed,
    )
    z = np.zeros((len(nodes), dim + 1), dtype=np.float32)
    for i, a in enumerate(adj):
        if a:
            z[i, :dim] = model.wv[str(i)]
        else:
            z[i, dim] = 1.0
    return FeatureBlock(
        name,
        z,
        [f"{name}_{i}" for i in range(dim)] + [f"{name}_isolated"],
        meta={"p": p, "q": q, "n_walks": len(walks), "isolated": int(z[:, dim].sum())},
    )


# ------------------------------------------------- assembly


def make_blocks(
    tg,
    graph: str,
    source: str,
    centrality: pd.DataFrame | None = None,
    svd_dim: int = 64,
    clmbr_dim: int | None = None,
    rw_steps: int = 8,
    n2v: dict | None = None,
    n2v_cache: dict | None = None,
    seed: int = 0,
) -> dict:
    """All feature blocks for one (graph condition, feature source) cell.
    `graph`: "safe" | "leaky" | "null" -- supplies structure, rwpe and n2v.
    `source`: "safe" | "leaky" -- which diagnosis vectors feed the diagnosis
    blocks. CLMBR and time depend on neither.
    `n2v`: keyword overrides for `node2vec_block`; `n2v_cache`: a dict reused
    across calls so each graph's embedding is trained once, not once per source."""
    matrix, codes = tg.vectors[source]
    g = tg.graphs[graph]
    n2v_kw = {"seed": seed, **(n2v or {})}
    if n2v_cache is not None and graph in n2v_cache:
        n2v_b = n2v_cache[graph]
    else:
        n2v_b = node2vec_block(g, tg.nodes, **n2v_kw)
        if n2v_cache is not None:
            n2v_cache[graph] = n2v_b
    blocks = [
        tfidf_svd_block(matrix, codes, svd_dim, seed),
        diagnosis_summary_block(matrix),
        time_block(tg.nodes),
        structure_block(g, tg.nodes, centrality),
        random_walk_pe_block(g, tg.nodes, rw_steps),
        n2v_b,
        clmbr_block(tg, clmbr_dim, seed),
    ]
    return {b.name: b for b in blocks if b is not None}


def assemble(blocks: dict, names: list):
    """Concatenate the named blocks (those that exist) into one matrix. Returns
    x (float32, UNSCALED), the column names, and a boolean mask of the columns
    that `standardize` should scale."""
    chosen = [blocks[n] for n in names if n in blocks]
    if not chosen:
        raise ValueError(f"none of {names} available (have {list(blocks)})")
    x = np.hstack([b.values for b in chosen]).astype(np.float32)
    columns = [c for b in chosen for c in b.columns]
    scale = np.concatenate([np.full(b.values.shape[1], b.scale) for b in chosen])
    return x, columns, scale


def build_pyg_data(
    tg,
    graph: str,
    source: str,
    feature_set: str,
    centrality: pd.DataFrame | None = None,
    drop_same_patient_edges: bool = False,
    blocks: dict | None = None,
    **block_kwargs,
):
    """One torch_geometric `Data` for a (graph, feature source, feature set) cell.

    Fields: x (unscaled), y, edge_index (both directions), edge_weight (the
    graph's 'weight' attribute, 1.0 if absent), edge_attr (1 = the edge joins two
    stays of the same patient), patient_idx (anonymous integer per patient, for
    grouped splits later), x_scale_mask (columns `standardize` will scale),
    feature_names, meta. No masks: splitting is a separate, later step.
    `drop_same_patient_edges=True` removes those edges (the paper found they
    carry much of the hypertension signal). Raw patient IDs are NOT stored; node
    i is tg.nodes[i]. Pass `blocks` to reuse precomputed blocks."""
    import torch
    from torch_geometric.data import Data

    if blocks is None:
        blocks = make_blocks(tg, graph, source, centrality, **block_kwargs)
    x, columns, scale = assemble(blocks, FEATURE_SETS[feature_set])
    y = np.array([tg.labels[n] for n in tg.nodes], dtype=np.int64)
    patient_idx, _ = pd.factorize(pd.Series([n[0] for n in tg.nodes]))

    idx = {n: i for i, n in enumerate(tg.nodes)}
    src, dst, w, same = [], [], [], []
    for u, v, d in tg.graphs[graph].edges(data=True):
        s = int(u[0] == v[0])
        if drop_same_patient_edges and s:
            continue
        for a, b in ((u, v), (v, u)):
            src.append(idx[a])
            dst.append(idx[b])
            w.append(float(d.get("weight", 1.0)))
            same.append(s)

    return Data(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_weight=torch.tensor(w, dtype=torch.float32),
        edge_attr=torch.tensor(same, dtype=torch.float32).unsqueeze(1),
        patient_idx=torch.from_numpy(patient_idx.astype(np.int64)),
        x_scale_mask=torch.from_numpy(scale),
        feature_names=columns,
        meta={"task": tg.task, "graph": graph, "source": source, "feature_set": feature_set},
    )


def standardize(data, fit_mask) -> None:
    """Standardise the scalable columns of data.x IN PLACE using the mean/std of
    the nodes in `fit_mask` (the training nodes) only. Call this after splits
    exist; applying it to all nodes would leak test statistics into x."""
    import torch

    cols = data.x_scale_mask
    fit = torch.as_tensor(fit_mask, dtype=torch.bool)
    ref = data.x[fit][:, cols]
    mean, std = ref.mean(0), ref.std(0).clamp_min(1e-8)
    data.x[:, cols] = (data.x[:, cols] - mean) / std


def check_data(data) -> None:
    """Cheap guards to run on every Data before training."""
    n = data.num_nodes
    assert data.x.shape[0] == n == data.y.shape[0] == data.patient_idx.shape[0]
    assert not np.isnan(data.x.numpy()).any(), "NaN in x"
    ei = data.edge_index
    if ei.numel():
        assert int(ei.max()) < n and int(ei.min()) >= 0
        fwd = set(zip(ei[0].tolist(), ei[1].tolist()))
        assert all((b, a) in fwd for a, b in fwd), "edge_index is not symmetric"
        assert not any(a == b for a, b in fwd), "self-loop in edge_index"


# ------------------------------------------------- importance diagnostics
# These use labels, on purpose, and are DESCRIPTIVE (like the paper's homophily,
# computed with all outcomes known). They are not the GNN's train/test split.


def _oof_auc(x, y, groups, seed, n_folds=5) -> float:
    """Out-of-fold AUROC of gradient-boosted trees, folds grouped by patient."""
    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in cv.split(x, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
        clf.fit(x[tr], y[tr])
        oof[te] = clf.predict_proba(x[te])[:, 1]
    return float(roc_auc_score(y, oof))


def block_ablation(blocks: dict, names: list, y: np.ndarray, groups: np.ndarray, seed: int = 0):
    """Which feature BLOCKS carry signal, with the graph left out entirely
    (trees on the node's own features, so it measures the features, not a GNN).
    Per block: AUROC of the block alone, and of everything else without it (a
    bigger drop = more important). AUROC is out-of-fold, folds grouped by
    `groups` (patient_idx). Importance by block, not per column: embedding
    columns are not individually meaningful."""
    names = [n for n in names if n in blocks]
    full = np.hstack([blocks[n].values for n in names])
    rows = [{"block": "ALL", "alone_auroc": _oof_auc(full, y, groups, seed), "drop_auroc": np.nan}]
    for n in names:
        alone = _oof_auc(blocks[n].values, y, groups, seed)
        rest = [m for m in names if m != n]
        drop = _oof_auc(np.hstack([blocks[m].values for m in rest]), y, groups, seed) if rest else np.nan
        rows.append({"block": n, "alone_auroc": alone, "drop_auroc": drop})
    out = pd.DataFrame(rows)
    out["auroc_lost_when_dropped"] = out.loc[0, "alone_auroc"] - out["drop_auroc"]
    return out


def column_auroc(block: FeatureBlock, y: np.ndarray, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Single-column AUROC (on `mask` nodes, default all), sorted by distance from 0.5."""
    mask = np.ones(len(y), dtype=bool) if mask is None else mask
    rows = [
        {"column": c, "auroc": roc_auc_score(y[mask], block.values[mask, j])}
        for j, c in enumerate(block.columns)
        if np.ptp(block.values[mask, j]) > 0
    ]
    out = pd.DataFrame(rows)
    return out.reindex((out["auroc"] - 0.5).abs().sort_values(ascending=False).index).reset_index(drop=True)


def code_gap_table(matrix, codes: list, y: np.ndarray, mask: np.ndarray | None = None, top: int = 20):
    """Per diagnosis code: share of positive vs. negative nodes that have it
    (over `mask` nodes, default all). The readable counterpart of the SVD block,
    and where a leak shows up by name (Hypertension: ICD-9 401.9, 67% of
    positives vs 5% of negatives)."""
    mask = np.ones(len(y), dtype=bool) if mask is None else mask
    present = sp.csr_matrix(matrix)[mask] > 0
    ym = y[mask]
    pos = np.asarray(present[ym == 1].mean(axis=0)).ravel()
    neg = np.asarray(present[ym == 0].mean(axis=0)).ravel()
    out = pd.DataFrame({"code": codes, "positive_rate": pos, "negative_rate": neg})
    out["gap"] = out["positive_rate"] - out["negative_rate"]
    return out.reindex(out["gap"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


def ablation_rows(table: pd.DataFrame, task: str, graph: str, source: str, variant: str) -> list:
    """`block_ablation` output as long-format aggregate rows (same schema as
    analyze_task's, so `write_aggregate` can take them). Check that function
    accepts these condition names; rename if it validates them."""
    rows = []
    for r in table.itertuples():
        for metric, value in (
            (f"block_alone_auroc_{r.block}", r.alone_auroc),
            (f"block_auroc_lost_when_dropped_{r.block}", r.auroc_lost_when_dropped),
        ):
            if pd.notna(value):
                rows.append(
                    {
                        "task": task,
                        "condition": f"{graph}_graph_{source}_features",
                        "graph_variant": variant,
                        "metric": metric,
                        "value": float(value),
                        "ci_low": None,
                        "ci_high": None,
                        "seed": None,
                    }
                )
    return rows