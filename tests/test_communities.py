import networkx as nx
import pytest

from ehrgraph.netstats.communities import compare_partitions, leiden_partition, louvain_partition


def _two_obvious_cliques():
    """Two dense, well-separated cliques connected by one weak bridge --
    both Louvain and Leiden should find exactly this same split."""
    graph = nx.Graph()
    clique_a = ["a1", "a2", "a3", "a4"]
    clique_b = ["b1", "b2", "b3", "b4"]
    for nodes in (clique_a, clique_b):
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                graph.add_edge(u, v, weight=1.0)
    graph.add_edge("a1", "b1", weight=0.01)  # weak bridge
    return graph


def test_louvain_partition_covers_every_node():
    graph = _two_obvious_cliques()
    partition = louvain_partition(graph)
    assert set(partition.keys()) == set(graph.nodes)


def test_leiden_partition_covers_every_node():
    graph = _two_obvious_cliques()
    partition = leiden_partition(graph)
    assert set(partition.keys()) == set(graph.nodes)


def test_leiden_partition_preserves_real_node_identity():
    graph = nx.Graph()
    graph.add_edge("patient_42", "patient_99", weight=1.0)
    partition = leiden_partition(graph)
    assert set(partition.keys()) == {"patient_42", "patient_99"}


def test_compare_partitions_agree_on_obvious_structure():
    graph = _two_obvious_cliques()
    result = compare_partitions(graph)
    assert result["adjusted_rand_index"] == pytest.approx(1.0)
    assert result["n_louvain_communities"] == 2
    assert result["n_leiden_communities"] == 2


def test_compare_partitions_returns_series_indexed_by_node():
    graph = _two_obvious_cliques()
    result = compare_partitions(graph)
    assert set(result["louvain"].index) == set(graph.nodes)
    assert set(result["leiden"].index) == set(graph.nodes)


def test_compare_partitions_reports_modularity_of_the_obvious_split():
    graph = _two_obvious_cliques()
    result = compare_partitions(graph)
    groups = [{"a1", "a2", "a3", "a4"}, {"b1", "b2", "b3", "b4"}]
    expected = nx.community.modularity(graph, groups, weight="weight")
    assert result["louvain_modularity"] == pytest.approx(expected)
    assert result["leiden_modularity"] == pytest.approx(expected)
    assert expected > 0.4


def test_leiden_partition_handles_tuple_node_ids():
    # Real graph nodes are (patient_id, prediction_time) tuples.
    graph = nx.relabel_nodes(_two_obvious_cliques(), lambda n: (hash(n) % 1000, n))
    partition = leiden_partition(graph)
    assert set(partition) == set(graph.nodes)
    assert len(set(partition.values())) == 2
