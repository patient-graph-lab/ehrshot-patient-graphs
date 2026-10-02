import networkx as nx
import numpy as np
import pytest

from ehrgraph.netstats.homophily import (
    class_insensitive_homophily,
    dirichlet_energy,
    edge_homophily,
)


def test_class_insensitive_homophily_perfect_segregation():
    # Two disjoint same-class components -- every edge is same-class.
    graph = nx.Graph()
    graph.add_edge("a1", "a2", weight=1.0)
    graph.add_edge("b1", "b2", weight=1.0)
    labels = {"a1": 0, "a2": 0, "b1": 1, "b2": 1}
    result = class_insensitive_homophily(graph, labels)
    assert result["per_class"][0] == pytest.approx(1.0)
    assert result["per_class"][1] == pytest.approx(1.0)
    assert result["class_insensitive_homophily"] == pytest.approx(1.0)


def test_class_insensitive_homophily_corrects_for_imbalance():
    # 8 class-0 nodes forming a clique, plus 2 class-1 nodes each connected
    # only to a class-0 node. Naive edge homophily is dominated by the
    # big clique (28/30); the class-insensitive score must not be.
    graph = nx.Graph()
    class0 = [f"c0_{i}" for i in range(8)]
    for i, u in enumerate(class0):
        for v in class0[i + 1 :]:
            graph.add_edge(u, v, weight=1.0)
    graph.add_edge("c1_0", "c0_0", weight=1.0)
    graph.add_edge("c1_1", "c0_1", weight=1.0)
    labels = {n: 0 for n in class0} | {"c1_0": 1, "c1_1": 1}

    result = class_insensitive_homophily(graph, labels)

    # class-0 neighbor slots: 28 internal edges x 2 + 2 bridge ends = 58,
    # of which 56 are same-class; class 1: 2 slots, 0 same-class.
    assert result["per_class"][0] == pytest.approx(56 / 58)
    assert result["per_class"][1] == pytest.approx(0.0)
    assert result["class_proportion"] == {0: pytest.approx(0.8), 1: pytest.approx(0.2)}
    # h_hat = max(0, 56/58 - 0.8) + max(0, 0 - 0.2), over C - 1 = 1
    assert result["class_insensitive_homophily"] == pytest.approx(56 / 58 - 0.8)
    assert edge_homophily(graph, labels) == pytest.approx(28 / 30)
    assert result["class_insensitive_homophily"] < edge_homophily(graph, labels)


def test_class_insensitive_homophily_is_near_zero_on_random_graph_at_any_prevalence():
    # The property the metric exists for: labels independent of wiring
    # -> ~0, whether prevalence is 0.5 or 0.14. (The Sep 25 first draft,
    # with no p_k baseline, scored ~0.37 here at prevalence 0.25.)
    rng = np.random.default_rng(0)
    graph = nx.gnp_random_graph(3000, 0.004, seed=0)
    for prevalence in (0.5, 0.25, 0.14):
        labels = {n: int(rng.random() < prevalence) for n in graph.nodes}
        h_hat = class_insensitive_homophily(graph, labels)["class_insensitive_homophily"]
        assert h_hat < 0.03


def test_class_insensitive_homophily_raises_on_unlabeled_node():
    graph = nx.Graph()
    graph.add_edge("a", "b")
    with pytest.raises(ValueError, match="no label"):
        class_insensitive_homophily(graph, {"a": 0})


def test_dirichlet_energy_lower_for_smooth_signal_than_bumpy_signal():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_edge("b", "c", weight=1.0)
    graph.add_edge("c", "a", weight=1.0)
    nodelist = ["a", "b", "c"]

    smooth = np.array([[1.0, 1.0], [1.01, 1.01], [0.99, 0.99]])
    bumpy = np.array([[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0]])

    smooth_energy = dirichlet_energy(graph, nodelist, smooth)
    bumpy_energy = dirichlet_energy(graph, nodelist, bumpy)
    assert smooth_energy < bumpy_energy


def test_dirichlet_energy_raises_on_length_mismatch():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1.0)
    with pytest.raises(ValueError, match="same length"):
        dirichlet_energy(graph, ["a", "b"], np.zeros((3, 2)))


def test_dirichlet_energy_raises_on_nodelist_not_matching_graph_nodes():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1.0)
    with pytest.raises(ValueError, match="exactly the graph's nodes"):
        dirichlet_energy(graph, ["a", "z"], np.zeros((2, 2)))
