import networkx as nx
import numpy as np
import pytest
from scipy.cluster.hierarchy import dendrogram, fcluster, is_valid_linkage

from ehrgraph.netstats.dendrogram import girvan_newman_tree, levels_to_linkage


def _two_cliques():
    graph = nx.Graph()
    for group in (["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]):
        for i, u in enumerate(group):
            for v in group[i + 1 :]:
                graph.add_edge(u, v, weight=5.0)
    graph.add_edge("a1", "b1", weight=1.5)
    return graph


def test_tree_is_a_valid_monotone_scipy_linkage_over_every_node():
    tree = girvan_newman_tree(_two_cliques())
    assert is_valid_linkage(tree.linkage)
    assert len(tree.leaves) == 8
    assert np.all(np.diff(tree.linkage[:, 2]) >= 0)  # heights never decrease going up
    assert tree.linkage[-1, 3] == 8  # the root holds every leaf
    assert len(dendrogram(tree.linkage, no_plot=True)["leaves"]) == 8
    assert tree.n_forced_joins == 0


def test_top_split_separates_the_two_cliques_and_is_the_most_modular_level():
    tree = girvan_newman_tree(_two_cliques())
    two = fcluster(tree.linkage, t=2, criterion="maxclust")
    groups = {}
    for leaf, cluster in zip(tree.leaves, two, strict=True):
        groups.setdefault(cluster, set()).add(leaf)
    assert sorted(map(sorted, groups.values())) == [
        ["a1", "a2", "a3", "a4"],
        ["b1", "b2", "b3", "b4"],
    ]
    assert len(tree.levels[tree.best_level]) == 2


def test_disconnected_components_are_joined_above_every_real_merge():
    graph = _two_cliques()
    graph.remove_edge("a1", "b1")
    graph.add_edge("x", "y", weight=2.0)  # a third, separate piece
    tree = girvan_newman_tree(graph)
    assert tree.n_forced_joins == 2
    assert is_valid_linkage(tree.linkage)
    real_top = tree.linkage[: -tree.n_forced_joins, 2].max()
    assert (tree.linkage[-tree.n_forced_joins :, 2] > real_top).all()


def test_levels_to_linkage_rejects_an_unfinished_sequence():
    levels = [(frozenset({"a", "b"}),)]
    with pytest.raises(ValueError, match="singletons"):
        levels_to_linkage(levels)
