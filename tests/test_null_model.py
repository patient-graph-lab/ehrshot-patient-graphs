import networkx as nx

from ehrgraph.graphs.null_model import build_null_graph


def _sample_graph() -> nx.Graph:
    # A small graph with enough edges for double_edge_swap to have real
    # room to work, plus one isolated node to confirm those stay isolated.
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    edges = [
        (0, 1, 0.9),
        (1, 2, 0.1),
        (2, 3, 0.5),
        (3, 4, 0.3),
        (4, 5, 0.7),
        (5, 6, 0.2),
        (6, 7, 0.6),
        (7, 8, 0.4),
        (8, 0, 0.8),
    ]
    for u, v, w in edges:
        graph.add_edge(u, v, weight=w)
    # node 9 stays isolated
    return graph


def test_null_graph_preserves_degree_sequence():
    graph = _sample_graph()
    null_graph = build_null_graph(graph, seed=0)
    real_degrees = sorted(d for _, d in graph.degree())
    null_degrees = sorted(d for _, d in null_graph.degree())
    assert real_degrees == null_degrees


def test_null_graph_preserves_isolated_nodes():
    graph = _sample_graph()
    null_graph = build_null_graph(graph, seed=0)
    assert null_graph.degree[9] == 0


def test_null_graph_preserves_edge_count():
    graph = _sample_graph()
    null_graph = build_null_graph(graph, seed=0)
    assert null_graph.number_of_edges() == graph.number_of_edges()


def test_null_graph_preserves_weight_multiset():
    graph = _sample_graph()
    null_graph = build_null_graph(graph, seed=0)
    real_weights = sorted(d["weight"] for _, _, d in graph.edges(data=True))
    null_weights = sorted(d["weight"] for _, _, d in null_graph.edges(data=True))
    assert real_weights == null_weights


def test_null_graph_edge_set_actually_changes():
    graph = _sample_graph()
    null_graph = build_null_graph(graph, seed=0)
    real_edges = {frozenset(e) for e in graph.edges()}
    null_edges = {frozenset(e) for e in null_graph.edges()}
    assert real_edges != null_edges


def test_null_graph_is_deterministic_given_a_seed():
    graph = _sample_graph()
    null_a = build_null_graph(graph, seed=42)
    null_b = build_null_graph(graph, seed=42)
    assert sorted(map(sorted, null_a.edges())) == sorted(map(sorted, null_b.edges()))
    weights_a = [null_a[u][v]["weight"] for u, v in sorted(map(tuple, null_a.edges()))]
    weights_b = [null_b[u][v]["weight"] for u, v in sorted(map(tuple, null_b.edges()))]
    assert weights_a == weights_b


def test_null_graph_handles_too_few_edges_to_swap():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3])
    graph.add_edge(1, 2, weight=0.5)  # only 1 edge -- nothing to swap
    null_graph = build_null_graph(graph, seed=0)
    assert null_graph.number_of_edges() == 1
    assert null_graph[1][2]["weight"] == 0.5
