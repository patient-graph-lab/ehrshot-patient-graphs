from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from ehrgraph.eval.aggregate import write_aggregate
from ehrgraph.netstats.task_analysis import (
    analyze_task,
    build_task_graphs,
    code_outcome_gaps,
    edge_set_stats,
    same_patient_stats,
)

T0 = datetime(2020, 1, 1)


def _node(pid, i=0):
    return (pid, T0 + timedelta(days=i))


def test_same_patient_stats_counts_own_admission_links():
    # Patient 1 has two admissions linked to each other (same outcome);
    # one cross-patient edge with a different outcome.
    a1, a2, b = _node(1, 0), _node(1, 30), _node(2)
    graph = nx.Graph([(a1, a2), (a2, b)])
    labels = {a1: 1, a2: 1, b: 0}
    stats = same_patient_stats(graph, labels)
    assert stats["same_patient_edge_fraction"] == pytest.approx(0.5)
    assert stats["same_patient_edge_same_outcome_rate"] == pytest.approx(1.0)
    # Without the same-patient edge, the only edge is cross-outcome -> 0.
    assert stats["class_insensitive_homophily_excl_same_patient"] == pytest.approx(0.0)


def test_edge_set_stats_splits_shared_and_condition_only_edges():
    a, b, c, d = _node(1), _node(2), _node(3), _node(4)
    safe = nx.Graph([(a, b), (b, c)])
    leaky = nx.Graph([(a, b), (c, d)])
    labels = {a: 1, b: 1, c: 0, d: 0}
    stats = edge_set_stats(safe, leaky, labels)
    assert stats["edge_jaccard"] == pytest.approx(1 / 3)
    assert stats["shared_edges"]["n_edges"] == 1
    assert stats["shared_edges"]["both_positive_rate"] == pytest.approx(1.0)
    assert stats["safe_only_edges"]["same_outcome_rate"] == pytest.approx(0.0)  # b-c
    assert stats["leaky_only_edges"]["same_outcome_rate"] == pytest.approx(1.0)  # c-d


def test_code_outcome_gaps_ranks_the_answer_key_code_first():
    # Code "leak" is present in every positive and no negative.
    y = np.array([1, 1, 0, 0])
    matrix = sp.csr_matrix(np.array([[1, 1], [1, 0], [0, 1], [0, 0]]))
    gaps = code_outcome_gaps(matrix, ["leak", "noise"], y)
    assert gaps.loc[0, "code"] == "leak"
    assert gaps.loc[0, "gap"] == pytest.approx(1.0)


def _synthetic_task(n_patients=60, seed=0):
    """Two outcome groups with different typical codes, some patients with
    two admissions -- enough structure for every metric to be defined."""
    rng = np.random.default_rng(seed)
    nodes, labels, rows = [], {}, []
    for pid in range(n_patients):
        y = int(pid % 3 == 0)
        for adm in range(1 + (pid % 4 == 0)):
            node = _node(pid, 40 * adm)
            nodes.append(node)
            labels[node] = y
            base = np.zeros(30)
            base[(0 if y else 15) + rng.choice(15, size=5, replace=False)] = 1
            base[rng.choice(30, size=2, replace=False)] = 1
            rows.append(base)
    safe = sp.csr_matrix(np.array(rows))
    leaky_rows = np.array(rows)
    leaky_rows[:, 29] = [labels[n] for n in nodes]  # code 29 = the answer key
    codes = [f"C{i}" for i in range(30)]
    features = rng.normal(size=(len(nodes), 8)) + np.array([[labels[n]] for n in nodes])
    return build_task_graphs(
        "toy_task",
        nodes,
        labels,
        (safe, codes),
        (sp.csr_matrix(leaky_rows), codes),
        features,
        n_null=3,
        k=5,
    )


def test_analyze_task_produces_valid_aggregate_rows(tmp_path):
    tg = _synthetic_task()
    analysis = analyze_task(tg, graph_variant="toy")
    out = write_aggregate(analysis.rows, tmp_path / "results" / "aggregate" / "toy.csv")
    df = pd.read_csv(out)

    assert set(df["condition"]) >= {
        "safe",
        "leaky",
        "null_model",
        "reference",
        "leaky_vs_safe",
        "shared_edges",
        "safe_only_edges",
        "leaky_only_edges",
    }
    for metric in (
        "class_insensitive_homophily",
        "class_insensitive_homophily_excl_same_patient",
        "same_patient_edge_fraction",
        "dirichlet_energy_clmbr",
        "leiden_modularity",
        "transitivity",
    ):
        assert set(df.loc[df["metric"] == metric, "condition"]) >= {"safe", "leaky", "null_model"}
    # Every defined null-ensemble summary carries its 95% interval; a metric
    # undefined on every seed (here: no same-patient edges left after
    # rewiring) is NaN with no interval, by design.
    null_rows = df[(df["condition"] == "null_model") & df["seed"].isna() & df["value"].notna()]
    assert null_rows[["ci_low", "ci_high"]].notna().all().all()

    gap = df.set_index(["condition", "metric"])["value"]
    assert gap[("leaky", "max_single_code_outcome_gap")] == pytest.approx(1.0)
    assert (
        gap[("leaky", "max_single_code_outcome_gap")] > gap[("safe", "max_single_code_outcome_gap")]
    )
    assert analysis.top_leaky_codes.loc[0, "code"] == "C29"
    # Planted outcome structure: the real graph is more homophilous than its nulls.
    assert (
        gap[("safe", "class_insensitive_homophily")]
        > gap[("null_model", "class_insensitive_homophily")]
    )


def test_build_task_graphs_rejects_misaligned_features():
    tg_nodes = [_node(1), _node(2)]
    vecs = (sp.csr_matrix(np.eye(2)), ["a", "b"])
    with pytest.raises(ValueError, match="one row per node"):
        build_task_graphs("t", tg_nodes, {n: 0 for n in tg_nodes}, vecs, vecs, np.zeros((3, 4)))
