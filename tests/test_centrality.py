import networkx as nx
import pytest

from ehrgraph.netstats.centrality import (
    add_distance_attribute,
    centrality_suite,
    degree_distribution,
)


def test_add_distance_attribute_is_one_minus_weight():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=0.7)
    add_distance_attribute(graph)
    assert graph["a"]["b"]["distance"] == pytest.approx(0.3)


def test_add_distance_attribute_clips_floating_point_overshoot():
    # A real, observed case: cosine similarity between near-identical
    # vectors can round to fractionally above 1.0, which would otherwise
    # make distance negative and break Dijkstra-based centrality.
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1.0000000000000004)
    add_distance_attribute(graph)
    assert graph["a"]["b"]["distance"] >= 0.0


def test_degree_distribution_matches_real_degrees():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_edge("a", "c", weight=1.0)
    dist = degree_distribution(graph)
    assert dist["a"] == 2
    assert dist["b"] == 1
    assert dist["c"] == 1


def _star_graph():
    # "center" connects to everyone; "leaf1"/"leaf2"/"leaf3" only connect
    # to center. Center should dominate every centrality measure.
    graph = nx.Graph()
    for leaf in ["leaf1", "leaf2", "leaf3"]:
        graph.add_edge("center", leaf, weight=0.9)
    return graph


def test_centrality_suite_center_of_star_dominates_all_four_measures():
    graph = _star_graph()
    suite = centrality_suite(graph, betweenness_k=10)
    assert set(suite.columns) == {"degree", "betweenness", "closeness", "eigenvector"}
    for column in suite.columns:
        assert suite.loc["center", column] >= suite.drop("center")[column].max()


def test_centrality_suite_does_not_crash_on_overshoot_weights():
    # Regression test for the exact real-data failure mode: an edge weight
    # fractionally above 1.0 must not raise "Contradictory paths found".
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1.0000000000000004)
    graph.add_edge("b", "c", weight=0.5)
    centrality_suite(graph, betweenness_k=10)  # should not raise


def test_centrality_suite_closeness_stays_bounded_on_disconnected_graph():
    # Regression test for a real observed bug: a tiny, tightly-knit
    # component (near-1.0 weights -> near-zero distances) produces a
    # numerically exploding closeness score if computed naively across a
    # disconnected graph (observed on real data: mean ~1.9e11). Closeness
    # must be restricted to the largest connected component, with NaN
    # (not a huge fabricated number) for everyone else.
    graph = nx.Graph()
    # the "main" component -- this is the largest, so closeness should be
    # computed here
    for i in range(5):
        graph.add_edge(f"main_{i}", f"main_{i + 1}", weight=0.5)
    # a tiny two-node component with a near-1.0 weight (near-zero distance)
    graph.add_edge("tiny_a", "tiny_b", weight=1.0 - 1e-15)

    suite = centrality_suite(graph, betweenness_k=10)
    main_closeness = suite.loc[[f"main_{i}" for i in range(6)], "closeness"]
    assert main_closeness.notna().all()
    assert (main_closeness.abs() < 10).all()  # sane, bounded values
    assert suite.loc["tiny_a", "closeness"] != suite.loc["tiny_a", "closeness"]  # NaN


def test_igraph_betweenness_and_closeness_match_networkx_exactly():
    # The igraph fast path must produce the same numbers networkx would.
    # Weights are continuous like real cosine similarities, plus some
    # identical-patient edges (1.0 and float overshoot, floored to
    # MIN_DISTANCE). Deliberately NOT coarse values like 0.2/0.5/0.8: those
    # create exact-length path ties where the libraries legitimately differ
    # -- igraph treats lengths within 1e-10 as tied, networkx needs exact
    # float equality, so 0.5 + 0.3 vs 0.8 counts as a tie in only one of them.
    graph = nx.connected_watts_strogatz_graph(120, 6, 0.2, seed=1)
    rng = __import__("random").Random(0)
    for u, v in graph.edges():
        roll = rng.random()
        graph[u][v]["weight"] = (
            1.0 if roll < 0.05 else 1.0000000000000004 if roll < 0.1 else rng.uniform(0.05, 0.99)
        )
    suite = centrality_suite(graph)

    add_distance_attribute(graph)
    expected_b = nx.betweenness_centrality(graph, weight="distance")
    expected_c = nx.closeness_centrality(graph, distance="distance")
    for node in graph.nodes:
        assert suite.loc[node, "betweenness"] == pytest.approx(expected_b[node], abs=1e-9)
        assert suite.loc[node, "closeness"] == pytest.approx(expected_c[node], rel=1e-9)


def test_eigenvector_restricted_to_largest_component():
    graph = nx.Graph()
    for i in range(5):
        graph.add_edge(f"main_{i}", f"main_{i + 1}", weight=0.5)
    graph.add_edge("tiny_a", "tiny_b", weight=0.9)
    suite = centrality_suite(graph)
    assert suite.loc[[f"main_{i}" for i in range(6)], "eigenvector"].gt(0.01).all()
    assert suite.loc[["tiny_a", "tiny_b"], "eigenvector"].isna().all()
