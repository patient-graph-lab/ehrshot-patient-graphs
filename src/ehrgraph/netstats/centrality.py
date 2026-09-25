"""Degree distribution + centrality suite (issue #11, PROJECT_GUIDE.md Module B).

**The correction this module exists to get right:** our edge weights are
similarities (bigger = more alike). `betweenness_centrality` and
`closeness_centrality` interpret a weight as a *distance* (bigger =
farther apart) -- passing similarity straight in silently inverts the
ranking, making the most-alike patients look the most distant. Convert
similarity to a distance first (`distance = 1 - weight`), and pass the
right attribute to each measure: distance-based measures get `distance`,
`eigenvector_centrality` wants raw `weight` (strength), unchanged.

**A second, easy-to-miss bug this module also guards against:** cosine
similarity between two *nearly identical* vectors can come out
fractionally above 1.0 (e.g. 1.0000000000000004) from ordinary floating-
point rounding -- confirmed against real data, not hypothetical (63 of
~18,700 edges in the real safe graph). That makes `distance = 1 - weight`
slightly negative, and Dijkstra-based shortest paths reject negative
weights outright (`ValueError: Contradictory paths found`). Weight is
clipped to `[0, 1]` before computing distance to prevent this.

**A third bug, also only visible by actually running this against real
data:** these graphs are not fully connected (the mutual k-NN sparsify
step leaves real isolated pockets, more so for the leaky condition --
see issue #6/#7/#8's fragmentation finding). `closeness_centrality` on a
disconnected graph is only computed within each node's own reachable
component, and a tiny tight-knit component where every edge happens to
have a near-zero distance (near-identical patients) produces a
near-zero *average* distance -- and closeness is defined as its
reciprocal, so it explodes numerically. Confirmed against real data: the
leaky graph's naive closeness came out with a mean of ~1.9 x 10^11 and a
max of ~1.4 x 10^13, meaningless by any reading. Closeness is instead
computed only within the largest connected component here (standard
practice for disconnected graphs); nodes outside it get `NaN`, reported
as "not applicable," not a fabricated number.

**Speed (Sep 25, measured on the real ~7,000-node Task A graphs):**
networkx's pure-Python Dijkstra took ~14 min for exact betweenness and
~8 min for closeness *per graph*. Both are now computed by igraph's C
implementation on the same `distance` attribute -- exact, and seconds per
graph -- then rescaled to networkx's normalization so the numbers mean the
same thing (verified against networkx in tests/test_centrality.py).
"""

from __future__ import annotations

import igraph as ig
import networkx as nx
import pandas as pd

#: Floor for `distance`. Identical-history patients have cosine similarity
#: 1.0 (distance 0), and igraph's betweenness requires strictly positive
#: lengths. 1e-6 is far below the smallest real gap between two distinct
#: patients yet well above igraph's 1e-10 tie tolerance.
MIN_DISTANCE = 1e-6


def add_distance_attribute(graph: nx.Graph) -> None:
    """Adds a `distance` edge attribute (1 - weight, floored at
    MIN_DISTANCE) in place -- the input to betweenness/closeness
    centrality, never `weight` directly."""
    for _, _, data in graph.edges(data=True):
        data["distance"] = max(MIN_DISTANCE, 1.0 - data["weight"])


def degree_distribution(graph: nx.Graph) -> pd.Series:
    """One row per node: its degree. Plot on a log-log axis to eyeball a
    power-law (real-world, hub-dominated) vs. Poisson-like (random-graph)
    shape -- see PROJECT_GUIDE.md's Module B intro."""
    return pd.Series(dict(graph.degree()), name="degree")


def _igraph_with_distance(graph: nx.Graph) -> tuple[ig.Graph, list]:
    order = list(graph.nodes)
    index = {n: i for i, n in enumerate(order)}
    edges = [(index[u], index[v]) for u, v in graph.edges()]
    ig_graph = ig.Graph(n=len(order), edges=edges)
    ig_graph.es["distance"] = [d["distance"] for _, _, d in graph.edges(data=True)]
    return ig_graph, order


def centrality_suite(
    graph: nx.Graph, betweenness_k: int | None = None, seed: int = 0
) -> pd.DataFrame:
    """One row per node, four centrality columns: degree, betweenness,
    closeness, eigenvector -- with the distance/strength semantics applied
    correctly (see module docstring).

    Betweenness is exact by default (igraph). `betweenness_k` below the
    node count switches to networkx's pivot-sampled approximation, kept
    only for parity with the original design; with igraph the exact
    version is cheap enough that there's no reason to sample.
    """
    add_distance_attribute(graph)
    n = graph.number_of_nodes()

    degree = nx.degree_centrality(graph)
    eigenvector = nx.eigenvector_centrality(graph, weight="weight", max_iter=1000)

    if betweenness_k is not None and betweenness_k < n:
        betweenness = nx.betweenness_centrality(
            graph, k=betweenness_k, weight="distance", seed=seed
        )
    else:
        ig_graph, order = _igraph_with_distance(graph)
        raw = ig_graph.betweenness(weights="distance", directed=False)
        # networkx's normalized undirected betweenness: divide by (n-1)(n-2)/2
        scale = 2.0 / ((n - 1) * (n - 2)) if n > 2 else 0.0
        betweenness = {node: b * scale for node, b in zip(order, raw, strict=True)}

    largest_cc = graph.subgraph(max(nx.connected_components(graph), key=len))
    ig_giant, giant_order = _igraph_with_distance(largest_cc)
    # On a connected graph igraph's normalized closeness, (n-1) / sum(dist),
    # is exactly networkx's closeness_centrality.
    giant_closeness = ig_giant.closeness(weights="distance", normalized=True)
    closeness_giant = dict(zip(giant_order, giant_closeness, strict=True))
    closeness = {node: closeness_giant.get(node, float("nan")) for node in graph.nodes}

    return pd.DataFrame(
        {
            "degree": degree,
            "betweenness": betweenness,
            "closeness": closeness,
            "eigenvector": eigenvector,
        }
    )
