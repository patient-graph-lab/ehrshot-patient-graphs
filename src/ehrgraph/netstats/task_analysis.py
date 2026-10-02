"""One task's full network analysis, as a library call (issues #11-#13, #15, #16).

Notebook 07 originally computed everything inline. Lifted here so Task A
(notebook 07), Task B (notebook 10) and Project 2's graph-variant sweep all
run the *same* code -- a copy-pasted notebook cell that drifts between tasks
would make the cross-task comparison (#17) meaningless.

Two steps:

  build_task_graphs(...)  -> TaskGraphs    the safe / leaky / null graphs,
                                           a null ensemble, centered CLMBR
                                           features, node labels
  analyze_task(tg)        -> TaskAnalysis  every metric, as long-format
                                           aggregate rows plus the in-memory
                                           per-node objects figures need

Per-node objects (centrality, partitions) stay in memory; only `rows` --
one number per (condition, metric) -- is meant for `write_aggregate`.

Null-model metrics are the mean over the ensemble with an empirical 95%
interval in `ci_low`/`ci_high`; per-node null figures use seed 0. The null
condition is written as `null_model` (pandas reads "null" as missing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score

from ehrgraph.graphs.null_model import build_null_graph
from ehrgraph.graphs.similarity import build_similarity_graph, graph_summary
from ehrgraph.graphs.vectorize import build_node_vectors
from ehrgraph.netstats.centrality import centrality_suite
from ehrgraph.netstats.communities import compare_partitions
from ehrgraph.netstats.homophily import (
    class_insensitive_homophily,
    dirichlet_energy,
    edge_homophily,
)
from ehrgraph.netstats.structure import structural_stats

#: The primary graph configuration (PROJECT_GUIDE.md Part 4 items 6 and 12):
#: diagnoses only, binary presence per node, TF-IDF + cosine, mutual k-NN, k=10,
#: built from the course OMOP tables' standard concepts with rebuilt labels
#: (team decision Oct 2 2026 -- every member can reproduce it).
PRIMARY_VARIANT = "omop_concept_binary_perpatient_tfidf_mutual_k10"
#: Same, on the original ICD billing codes (keeps the 2015 coding-era split).
SOURCE_CODES_VARIANT = "omop_source_binary_perpatient_tfidf_mutual_k10"
#: Same as primary, but with Stanford's official labels (full release needed).
OFFICIAL_LABELS_VARIANT = "omop_concept_binary_perpatient_tfidf_mutual_k10_official_labels"
MEASURES = ["degree", "betweenness", "closeness", "eigenvector"]
COMMUNITY_METRICS = [
    "adjusted_rand_index",
    "n_louvain_communities",
    "n_leiden_communities",
    "louvain_modularity",
    "leiden_modularity",
]


@dataclass
class TaskGraphs:
    task: str
    nodes: list  # (patient_id, prediction_time)
    labels: dict  # node -> 0/1
    graphs: dict  # "safe", "leaky", "null" (= nulls[0])
    nulls: list
    features: np.ndarray | None  # centered CLMBR embeddings, rows = feature_nodes; None if absent
    vectors: dict = field(default_factory=dict)  # "safe"/"leaky" -> (matrix, codes)
    feature_nodes: list = field(default_factory=list)  # nodes that have a CLMBR fingerprint

    @property
    def prevalence(self) -> float:
        return float(np.mean(list(self.labels.values())))


@dataclass
class TaskAnalysis:
    rows: list
    struct: dict
    centrality: dict
    communities: dict
    community_outcome: dict
    homophily: dict
    null_homophily: list
    edge_sets: dict
    top_leaky_codes: pd.DataFrame


def build_task_graphs(
    task: str,
    nodes: list,
    labels: dict,
    safe_vectors: tuple[sp.spmatrix, list],
    leaky_vectors: tuple[sp.spmatrix, list],
    features: np.ndarray | None,
    n_null: int = 10,
    k: int = 10,
    feature_nodes: list | None = None,
) -> TaskGraphs:
    """Pure graph construction from precomputed vectors -- no data access,
    so tests can drive it with synthetic inputs.

    `features` (CLMBR fingerprints) is optional: None when the full release
    isn't on this machine, and then Dirichlet energy is skipped. Its rows are
    `feature_nodes` (default: all `nodes`, in order) -- rebuilt labels can
    include a few stays Stanford never fingerprinted. It is centered here (an
    offset shared by every patient isn't similarity, and would lower every
    graph's Dirichlet energy by the same amount)."""
    feature_nodes = list(nodes if feature_nodes is None else feature_nodes)
    if features is not None and features.shape[0] != len(feature_nodes):
        raise ValueError("features must have one row per node, in `nodes` order.")
    safe = build_similarity_graph(safe_vectors[0], nodes, k=k)
    leaky = build_similarity_graph(leaky_vectors[0], nodes, k=k)
    nulls = [build_null_graph(safe, seed=s) for s in range(n_null)]
    centered = None
    if features is not None:
        centered = np.asarray(features, dtype=np.float64)
        centered = centered - centered.mean(axis=0)
    return TaskGraphs(
        task=task,
        nodes=list(nodes),
        labels=dict(labels),
        graphs={"safe": safe, "leaky": leaky, "null": nulls[0]},
        nulls=nulls,
        features=centered,
        vectors={"safe": safe_vectors, "leaky": leaky_vectors},
        feature_nodes=feature_nodes if features is not None else [],
    )


def task_graphs_from_labels(
    db,
    labels_df: pd.DataFrame,
    task: str,
    clmbr_path: Path | None = None,
    n_null: int = 10,
    k: int = 10,
    table: str = "condition",
    exclude: set | None = None,
) -> TaskGraphs:
    """Graphs for one task from any label table (patient_id,
    prediction_time, value) and any meds_reader-shaped `db` -- the course
    OMOP tables (`ehrgraph.data.omop.OmopDatabase`, primary) or the full
    release's meds_reader database.

    `clmbr_path`: the full release's clmbr_features.pkl, or None. Nodes
    without a fingerprint are kept in every graph; Dirichlet energy is then
    measured on the fingerprinted ones only. `exclude`: nodes to drop before
    anything is built (sensitivity analyses)."""
    labels_df = labels_df[labels_df["patient_id"].isin(set(db))].reset_index(drop=True)
    nodes = list(zip(labels_df["patient_id"], labels_df["prediction_time"], strict=True))
    if exclude:
        keep = [n not in exclude for n in nodes]
        labels_df = labels_df[keep].reset_index(drop=True)
        nodes = [n for n, k_ in zip(nodes, keep, strict=True) if k_]
    labels = dict(zip(nodes, labels_df["value"].astype(int), strict=True))
    if len(labels) != len(nodes):
        raise ValueError("duplicate (patient_id, prediction_time) nodes in the labels")

    safe_m, _, safe_codes = build_node_vectors(db, nodes, table=table, censor=True)
    leaky_m, _, leaky_codes = build_node_vectors(db, nodes, table=table, censor=False)

    features, feature_nodes = None, None
    if clmbr_path is not None:
        from ehrgraph.eval.baseline import load_clmbr_features

        feats = load_clmbr_features(Path(clmbr_path))
        feats = feats.set_index(["patient_id", "prediction_time"])["embedding"]
        feature_nodes = [n for n in nodes if n in feats.index]
        features = np.stack([feats.loc[n] for n in feature_nodes]) if feature_nodes else None

    return build_task_graphs(
        task,
        nodes,
        labels,
        (safe_m, safe_codes),
        (leaky_m, leaky_codes),
        features,
        n_null,
        k,
        feature_nodes=feature_nodes,
    )


def load_task_graphs(
    db,
    assets: Path,
    task: str,
    n_null: int = 10,
    k: int = 10,
    table: str = "condition",
    exclude: set | None = None,
) -> TaskGraphs:
    """Full-release path: Stanford's official labels and CLMBR fingerprints
    from EHRSHOT_ASSETS (needs your own full-release download). `db` may be
    the meds_reader database or a `vocabulary.MappedConditions` view."""
    labels_df = pd.read_csv(
        Path(assets) / "benchmark" / task / "labeled_patients.csv", parse_dates=["prediction_time"]
    )
    return task_graphs_from_labels(
        db,
        labels_df,
        task,
        clmbr_path=Path(assets) / "features" / "clmbr_features.pkl",
        n_null=n_null,
        k=k,
        table=table,
        exclude=exclude,
    )


def _same_patient(u, v) -> bool:
    return u[0] == v[0]


def same_patient_stats(graph: nx.Graph, labels: dict) -> dict:
    """How much of the graph is a patient linked to their own other
    admissions -- 37.5% of Task A's safe edges (found Sep 25). Those edges
    share an outcome far more often than chance, so homophily is reported
    with and without them."""
    same = [(u, v) for u, v in graph.edges() if _same_patient(u, v)]
    m = graph.number_of_edges()
    without = graph.copy()
    without.remove_edges_from(same)
    return {
        "same_patient_edge_fraction": len(same) / m if m else float("nan"),
        "same_patient_edge_same_outcome_rate": (
            float(np.mean([labels[u] == labels[v] for u, v in same])) if same else float("nan")
        ),
        "class_insensitive_homophily_excl_same_patient": class_insensitive_homophily(
            without, labels
        )["class_insensitive_homophily"],
    }


def edge_set_stats(safe: nx.Graph, leaky: nx.Graph, labels: dict) -> dict:
    """Which links leakage adds and removes, and whether the added ones join
    same-outcome patients more often than the removed ones (issue #15)."""
    s = {frozenset(e) for e in safe.edges()}
    lk = {frozenset(e) for e in leaky.edges()}
    sets = {"shared_edges": s & lk, "safe_only_edges": s - lk, "leaky_only_edges": lk - s}
    out = {"edge_jaccard": len(s & lk) / len(s | lk) if s | lk else float("nan")}
    for name, edges in sets.items():
        pairs = [tuple(e) for e in edges]
        same_outcome = [labels[u] == labels[v] for u, v in pairs]
        both_pos = [labels[u] == 1 and labels[v] == 1 for u, v in pairs]
        out[name] = {
            "n_edges": len(pairs),
            "same_outcome_rate": float(np.mean(same_outcome)) if pairs else float("nan"),
            "both_positive_rate": float(np.mean(both_pos)) if pairs else float("nan"),
            "same_patient_fraction": (
                float(np.mean([_same_patient(u, v) for u, v in pairs])) if pairs else float("nan")
            ),
        }
    return out


def code_outcome_gaps(matrix: sp.spmatrix, codes: list, y: np.ndarray) -> pd.DataFrame:
    """Per code: share of positive vs. negative nodes whose vector has it.
    A huge gap in the leaky vectors means the future record contains the
    answer (Hypertension: ICD-9 401.9 in 67% of positives vs 5% of
    negatives, found Sep 25)."""
    present = sp.csc_matrix(matrix) > 0
    pos = np.asarray(present[y == 1].mean(axis=0)).ravel()
    neg = np.asarray(present[y == 0].mean(axis=0)).ravel()
    return (
        pd.DataFrame({"code": codes, "positive_rate": pos, "negative_rate": neg})
        .assign(gap=lambda d: d["positive_rate"] - d["negative_rate"])
        .sort_values("gap", ascending=False)
        .reset_index(drop=True)
    )


def _homophily_metrics(
    graph: nx.Graph, labels: dict, feature_nodes: list, features: np.ndarray | None
) -> dict:
    ci = class_insensitive_homophily(graph, labels)
    out = {
        "edge_homophily": edge_homophily(graph, labels),
        "class_insensitive_homophily": ci["class_insensitive_homophily"],
        "same_outcome_rate_positive": ci["per_class"][1],
        "same_outcome_rate_negative": ci["per_class"][0],
    }
    if features is not None:
        # on the fingerprinted nodes only (all of them with official labels)
        sub = graph.subgraph(feature_nodes)
        out["dirichlet_energy_clmbr"] = dirichlet_energy(sub, feature_nodes, features)
    return out | same_patient_stats(graph, labels)


def _community_outcome(partition: pd.Series, y: pd.Series, min_size: int) -> dict:
    df = pd.DataFrame({"community": partition, "y": y.loc[partition.index]})
    sizes = df["community"].map(df["community"].value_counts())
    big = df[sizes >= min_size]
    rates = big.groupby("community")["y"].agg(["mean", "size"])
    grand = big["y"].mean()
    between = (rates["size"] * (rates["mean"] - grand) ** 2).sum()
    total = ((big["y"] - grand) ** 2).sum()
    return {
        "rates": rates,
        "eta_squared": between / total if total else float("nan"),
        "nmi": normalized_mutual_info_score(big["y"], big["community"]),
        "share_in_reported_communities": len(big) / len(df),
    }


def analyze_task(
    tg: TaskGraphs, graph_variant: str = PRIMARY_VARIANT, min_community: int = 20
) -> TaskAnalysis:
    """Every Project 1 metric for one task. `min_community`: Leiden groups
    smaller than this are left out of outcome-rate summaries (noisy, and a
    tiny group's rate says too much about the few patients in it)."""
    rows: list[dict] = []

    def add(condition, metric, value, ci_low=None, ci_high=None, seed=None):
        rows.append(
            {
                "task": tg.task,
                "condition": "null_model" if condition == "null" else condition,
                "graph_variant": graph_variant,
                "metric": metric,
                "value": float(value),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "seed": seed,
            }
        )

    def add_null(metric, values):
        # Seeds where a metric is undefined (e.g. a rewired graph with no
        # same-patient edges left) are skipped rather than poisoning the mean.
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            add("null", metric, float("nan"))
            return
        lo, hi = np.percentile(values, [2.5, 97.5])
        add("null", metric, values.mean(), lo, hi)

    real = ("safe", "leaky")
    labels, y = tg.labels, pd.Series(tg.labels)

    # Structure (#11)
    struct = {n: structural_stats(tg.graphs[n]) | graph_summary(tg.graphs[n]) for n in real}
    null_struct = [structural_stats(g) | graph_summary(g) for g in tg.nulls]
    for n in real:
        for metric, value in struct[n].items():
            add(n, metric, value)
    for metric in null_struct[0]:
        add_null(metric, [s[metric] for s in null_struct])
    struct["null"] = null_struct

    # Centrality (#11) -- per node, null seed 0
    centrality = {n: centrality_suite(tg.graphs[n]) for n in ("safe", "leaky", "null")}
    for n, c in centrality.items():
        seed = 0 if n == "null" else None
        for measure in MEASURES:
            col = c[measure].dropna()
            for stat, value in (
                ("mean", col.mean()),
                ("median", col.median()),
                ("p90", col.quantile(0.9)),
                ("max", col.max()),
            ):
                add(n, f"{measure}_centrality_{stat}", value, seed=seed)
            y_col = y.loc[col.index]
            auc = roc_auc_score(y_col, col) if y_col.nunique() == 2 else float("nan")
            add(n, f"{measure}_centrality_auroc_vs_label", auc, seed=seed)
        rho = c[MEASURES].corr(method="spearman")
        for i, a in enumerate(MEASURES):
            for b in MEASURES[i + 1 :]:
                add(n, f"spearman_{a}_vs_{b}", rho.loc[a, b], seed=seed)

    # Communities (#12)
    communities = {n: compare_partitions(tg.graphs[n], seed=0) for n in real}
    null_comm = [compare_partitions(g, seed=0) for g in tg.nulls]
    communities["null"] = null_comm[0]
    outcome = {
        n: _community_outcome(communities[n]["leiden"], y, min_community)
        for n in ("safe", "leaky", "null")
    }
    null_outcome = [_community_outcome(r["leiden"], y, min_community) for r in null_comm]
    for n in real:
        for m in COMMUNITY_METRICS:
            add(n, m, communities[n][m])
        for m in ("eta_squared", "nmi", "share_in_reported_communities"):
            add(n, f"leiden_outcome_{m}", outcome[n][m])
    for m in COMMUNITY_METRICS:
        add_null(m, [r[m] for r in null_comm])
    for m in ("eta_squared", "nmi", "share_in_reported_communities"):
        add_null(f"leiden_outcome_{m}", [o[m] for o in null_outcome])

    # Homophily, Dirichlet energy, same-patient links (#13, #15)
    homophily = {
        n: _homophily_metrics(tg.graphs[n], labels, tg.feature_nodes, tg.features) for n in real
    }
    null_h = [_homophily_metrics(g, labels, tg.feature_nodes, tg.features) for g in tg.nulls]
    for n in real:
        for metric, value in homophily[n].items():
            add(n, metric, value)
    for metric in null_h[0]:
        add_null(metric, [r[metric] for r in null_h])
    add("reference", "prevalence", tg.prevalence)
    add("reference", "n_labels", len(tg.nodes))
    add("reference", "clmbr_coverage", len(tg.feature_nodes) / len(tg.nodes))

    # Which links leakage changes (#15)
    edge_sets = edge_set_stats(tg.graphs["safe"], tg.graphs["leaky"], labels)
    add("leaky_vs_safe", "edge_jaccard", edge_sets["edge_jaccard"])
    for name in ("shared_edges", "safe_only_edges", "leaky_only_edges"):
        for metric, value in edge_sets[name].items():
            add(name, metric, value)

    # How strongly any single code gives the outcome away (#15)
    y_arr = np.array([labels[n] for n in tg.nodes])
    gaps = {}
    for n in real:
        if n in tg.vectors:
            gaps[n] = code_outcome_gaps(*tg.vectors[n], y_arr)
            add(n, "max_single_code_outcome_gap", gaps[n]["gap"].iloc[0])
    top_leaky = gaps.get("leaky", pd.DataFrame()).head(10)

    return TaskAnalysis(
        rows=rows,
        struct=struct,
        centrality=centrality,
        communities=communities,
        community_outcome=outcome,
        homophily=homophily,
        null_homophily=null_h,
        edge_sets=edge_sets,
        top_leaky_codes=top_leaky,
    )


def summary_table(rows: list, metrics: list, conditions=("safe", "leaky", "null_model")):
    """Display helper: one row per metric, one column per condition, plus the
    null ensemble's 95% range as text."""
    df = pd.DataFrame(rows)
    df = df[df["metric"].isin(metrics) & df["condition"].isin(conditions) & df["seed"].isna()]
    table = df.pivot(index="metric", columns="condition", values="value").reindex(metrics)
    table = table[[c for c in conditions if c in table.columns]].astype(object)
    if "null_model" in conditions:
        null = df[df["condition"] == "null_model"].set_index("metric")
        table["null 95% range"] = [
            f"{null.loc[m, 'ci_low']:.4g} – {null.loc[m, 'ci_high']:.4g}"
            if m in null.index and pd.notna(null.loc[m, "ci_low"])
            else ""
            for m in table.index
        ]
    return table


def coding_era(matrix: sp.spmatrix, codes: list) -> np.ndarray:
    """Per node: "icd9" / "icd10" if its vector holds codes from only one
    ICD revision, "mixed" if both, "none" if neither. Use RAW-code vectors:
    after SNOMED harmonization the revision is no longer visible."""
    present = sp.csr_matrix(matrix) > 0
    is9 = np.array([c.startswith("ICD9") for c in codes])
    is10 = np.array([c.startswith("ICD10") for c in codes])
    n9 = (
        np.asarray(present[:, is9].sum(axis=1)).ravel() if is9.any() else np.zeros(present.shape[0])
    )
    n10 = (
        np.asarray(present[:, is10].sum(axis=1)).ravel()
        if is10.any()
        else np.zeros(present.shape[0])
    )
    return np.where(
        (n9 > 0) & (n10 > 0), "mixed", np.where(n9 > 0, "icd9", np.where(n10 > 0, "icd10", "none"))
    )


def era_stats(
    graph: nx.Graph, era: dict, labels: dict, n_shuffles: int = 20, seed: int = 0
) -> dict:
    """How strongly a graph is split by ICD revision, and how much of its
    homophily that split could explain. The latter shuffles outcomes
    *within* each era (keeping every era's outcome rate) and re-measures
    homophily: what's left is what era alone would produce."""
    m = graph.number_of_edges()
    cross = sum(1 for u, v in graph.edges() if {era[u], era[v]} == {"icd9", "icd10"})
    g = graph.copy()
    nx.set_node_attributes(g, era, "era")
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes)
    by_era: dict = {}
    for n in nodes:
        by_era.setdefault(era[n], []).append(n)
    shuffled = []
    for _ in range(n_shuffles):
        permuted = {}
        for members in by_era.values():
            values = rng.permutation([labels[n] for n in members])
            permuted.update(zip(members, values, strict=True))
        shuffled.append(class_insensitive_homophily(graph, permuted)["class_insensitive_homophily"])
    lo, hi = np.percentile(shuffled, [2.5, 97.5])
    return {
        "icd9_icd10_cross_edges": float(cross),
        "icd9_icd10_cross_edge_fraction": cross / m if m else float("nan"),
        "era_assortativity": nx.attribute_assortativity_coefficient(g, "era"),
        "homophily_era_only": float(np.mean(shuffled)),
        "homophily_era_only_lo": float(lo),
        "homophily_era_only_hi": float(hi),
    }
