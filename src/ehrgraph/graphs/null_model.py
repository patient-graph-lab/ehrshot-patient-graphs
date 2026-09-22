"""The null graph condition (issue #8, PROJECT_GUIDE.md Module B).

A positive/noise control: if a graph "helps" prediction mainly because
*any* graph-shaped structure would help (rather than because this graph
captures real medical similarity), the null model should help about as
much as the real graphs do. If it doesn't, that's evidence the real
graphs' structure -- not just their existence -- is doing the work.

**Correction this implementation follows (PROJECT_GUIDE.md Module B):**
`nx.configuration_model` is NOT used here -- it silently discards edge
weights, which would handicap the "control" by comparing a weighted real
graph against an unweighted fake one; any gain from the real graph could
then just be an artifact of weight vs. no-weight, not real vs. null
structure. Instead: `nx.double_edge_swap` keeps every node's degree
*exactly* what it was in the real graph (a real, structural invariant),
and the real graph's own edge-weight multiset is reassigned to the
swapped edges at random afterward -- so the null graph has the same
degree sequence and the same weight distribution as the real graph, just
scrambled about who-connects-to-whom.
"""

from __future__ import annotations

import random

import networkx as nx


def build_null_graph(
    graph: nx.Graph,
    n_swaps_per_edge: int = 10,
    max_tries_multiplier: int = 100,
    seed: int | None = None,
) -> nx.Graph:
    """A degree-preserving, weight-shuffled null version of `graph`.

    Isolated nodes (degree 0) are unaffected -- double_edge_swap only
    touches the edge set, and an isolated node has no edges to touch, so
    it stays isolated in the null graph too (its degree, 0, is still
    exactly preserved).
    """
    null_graph = graph.copy()
    n_edges = null_graph.number_of_edges()
    if n_edges >= 2:
        n_swaps = n_edges * n_swaps_per_edge
        max_tries = n_swaps * max_tries_multiplier
        nx.double_edge_swap(null_graph, nswap=n_swaps, max_tries=max_tries, seed=seed)

    weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    rng = random.Random(seed)
    rng.shuffle(weights)
    for (u, v), weight in zip(null_graph.edges(), weights, strict=True):
        null_graph[u][v]["weight"] = weight

    return null_graph
