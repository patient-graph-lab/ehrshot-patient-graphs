"""Whole-graph structural statistics (issue #11, PROJECT_GUIDE.md Module B).

One number per graph, not one per node -- so every value here is safe to
write straight to `results/aggregate/`. Complements `centrality_suite`
(per-node, stays local) and `similarity.graph_summary` (components).

Path-based numbers (average shortest path, diameter) are hop counts in
the largest connected component: on a disconnected graph "average
distance" is undefined across components, and hops rather than
`1 - similarity` distances keep them comparable to the textbook
small-world figures in Lecture 2.

Average path length is *estimated* from BFS out of a random sample of
source nodes (default 500). The exact all-pairs version took ~190 s per
graph on the real ~7,000-node Task A graphs (measured Sep 25), which is
~40 min across the safe/leaky graphs plus a 10-seed null ensemble; the
sample mean is an unbiased estimate of the same quantity. Pass
`path_sources=None` for the exact value.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def average_path_length_sampled(
    graph: nx.Graph, n_sources: int | None = 500, seed: int = 0
) -> float:
    """Mean hop distance between distinct node pairs of a connected graph,
    via BFS from `n_sources` randomly chosen sources (all nodes if None
    or if the graph is smaller)."""
    nodes = list(graph.nodes)
    if n_sources is not None and n_sources < len(nodes):
        rng = np.random.default_rng(seed)
        nodes = [nodes[i] for i in rng.choice(len(nodes), size=n_sources, replace=False)]
    total, pairs = 0, 0
    for source in nodes:
        lengths = nx.single_source_shortest_path_length(graph, source)
        total += sum(lengths.values())
        pairs += len(lengths) - 1
    return total / pairs


def structural_stats(graph: nx.Graph, path_sources: int | None = 500, seed: int = 0) -> dict:
    """Density, degree summary, clustering, assortativity, and LCC path
    lengths for one graph."""
    degrees = np.array([d for _, d in graph.degree()], dtype=float)
    largest_cc = graph.subgraph(max(nx.connected_components(graph), key=len))

    return {
        "density": nx.density(graph),
        "mean_degree": float(degrees.mean()),
        "median_degree": float(np.median(degrees)),
        "max_degree": float(degrees.max()),
        # Unweighted and weighted clustering answer slightly different
        # questions -- "are my neighbors linked?" vs. "how strongly?" --
        # both reported; triadic closure (Module C) is the unweighted one.
        "avg_clustering": nx.average_clustering(graph),
        "avg_clustering_weighted": nx.average_clustering(graph, weight="weight"),
        "transitivity": nx.transitivity(graph),
        "degree_assortativity": nx.degree_assortativity_coefficient(graph),
        "lcc_avg_shortest_path_hops": average_path_length_sampled(
            largest_cc, n_sources=path_sources, seed=seed
        ),
        # 2-sweep lower bound: exact diameter is another all-pairs pass for
        # a descriptive number, and the bound is tight on graphs like these.
        "lcc_diameter_hops_lower_bound": float(nx.approximation.diameter(largest_cc, seed=0)),
    }
