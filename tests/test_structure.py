import networkx as nx
import pytest

from ehrgraph.netstats.structure import average_path_length_sampled, structural_stats


def test_structural_stats_on_known_graph():
    # A 4-cycle with one chord (a-c) plus an isolated node e: the LCC is
    # a,b,c,d; e counts toward density/degree but not path lengths.
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [("a", "b", 1.0), ("b", "c", 1.0), ("c", "d", 1.0), ("d", "a", 1.0), ("a", "c", 1.0)]
    )
    graph.add_node("e")
    stats = structural_stats(graph)

    assert stats["density"] == pytest.approx(5 / 10)
    assert stats["mean_degree"] == pytest.approx(10 / 5)
    assert stats["max_degree"] == 3
    # 2 triangles; transitivity = 3 * 2 / connected triples (a:3, c:3, b:1, d:1)
    assert stats["transitivity"] == pytest.approx(6 / 8)
    # LCC pairs: 5 at distance 1, b-d at distance 2 -> 7 / 6
    assert stats["lcc_avg_shortest_path_hops"] == pytest.approx(7 / 6)
    assert stats["lcc_diameter_hops_lower_bound"] == 2


def test_sampled_path_length_matches_exact_when_sampling_everything_and_is_close_otherwise():
    graph = nx.connected_watts_strogatz_graph(400, 6, 0.1, seed=0)
    exact = nx.average_shortest_path_length(graph)
    assert average_path_length_sampled(graph, n_sources=None) == pytest.approx(exact)
    assert average_path_length_sampled(graph, n_sources=100) == pytest.approx(exact, rel=0.03)
