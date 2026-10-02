"""Girvan-Newman as a real dendrogram (issue #14, PROJECT_GUIDE.md Module C).

`networkx.algorithms.community.girvan_newman` yields a *sequence of
partitions*, each with one more community than the last, as the edge with
the highest betweenness is repeatedly removed. Calling `next()` once gives
only the first split into two -- the guide's original bug. A dendrogram
needs the whole sequence, turned into the bottom-up merge table
(`scipy` linkage matrix) that `scipy.cluster.hierarchy.dendrogram` draws.

The conversion, read bottom-up: the last partition is all singletons;
stepping back one level, exactly one community is the union of two
communities from the level below -- that union is one merge. The merge
height is how far up the sequence it happens, so the first split
Girvan-Newman makes (the most "bridge-like" edge) is the top of the tree.

A graph that starts disconnected has no Girvan-Newman merge joining its
components. Those top-level joins are added at one height above every
real merge and reported (`n_forced_joins`), so the plot can show where
the algorithm's evidence stops.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from networkx.algorithms.community import girvan_newman


@dataclass
class GirvanNewmanTree:
    levels: list  # partitions, coarsest (connected components) first
    leaves: list  # node order used for linkage rows 0..n-1
    linkage: np.ndarray  # scipy linkage matrix, shape (n-1, 4)
    n_forced_joins: int  # top-level joins between originally disconnected components
    modularity: list  # weighted modularity of each level
    best_level: int  # index into `levels` with the highest modularity


def girvan_newman_levels(graph: nx.Graph, weight: str | None = "weight") -> list:
    """Every Girvan-Newman level, starting from the graph's own connected
    components. With `weight`, edge betweenness uses `1 / weight` as a
    length: our weights are relative risks (bigger = stronger link), and
    shortest paths need "bigger = farther" -- the same similarity-vs-
    distance trap as Module B's centrality correction."""
    work = graph.copy()
    if weight is not None:
        for _, _, data in work.edges(data=True):
            data["_length"] = 1.0 / data[weight]

        def most_valuable_edge(g):
            centrality = nx.edge_betweenness_centrality(g, weight="_length")
            return max(centrality, key=centrality.get)
    else:
        most_valuable_edge = None

    levels = [tuple(frozenset(c) for c in nx.connected_components(work))]
    for partition in girvan_newman(work, most_valuable_edge=most_valuable_edge):
        levels.append(tuple(frozenset(c) for c in partition))
    return levels


def levels_to_linkage(levels: list) -> tuple[np.ndarray, list, int]:
    """Bottom-up scipy linkage matrix from top-down Girvan-Newman levels.

    Row t of the result merges clusters Z[t,0] and Z[t,1] at height Z[t,2]
    into new cluster n+t with Z[t,3] leaves -- scipy's convention. Leaves
    are numbered by the returned `leaves` order.
    """
    finest = levels[-1]
    if any(len(c) != 1 for c in finest):
        raise ValueError("the last level must be all singletons (run Girvan-Newman to the end)")
    leaves = sorted((next(iter(c)) for c in finest), key=str)
    n = len(leaves)
    cluster_id = {frozenset([leaf]): i for i, leaf in enumerate(leaves)}
    rows = []
    next_id = n
    height = 0.0

    for coarser, finer in zip(reversed(levels[:-1]), reversed(levels[1:]), strict=True):
        new = set(coarser) - set(finer)
        if len(new) != 1 or len(finer) - len(coarser) != 1:
            raise ValueError("each Girvan-Newman level must split exactly one community in two")
        merged = new.pop()
        parts = [c for c in finer if c <= merged]
        height += 1.0
        a, b = sorted(cluster_id[p] for p in parts)
        rows.append([a, b, height, len(merged)])
        cluster_id[merged] = next_id
        next_id += 1

    roots = [cluster_id[c] for c in levels[0]]
    sizes = {cluster_id[c]: len(c) for c in levels[0]}
    n_forced = len(roots) - 1
    top = height + 1.0
    # Join disconnected components, largest first, all at one height above
    # the last real merge -- no evidence orders them, so none is implied.
    roots.sort(key=lambda r: -sizes[r])
    current, current_size = roots[0], sizes[roots[0]]
    for other in roots[1:]:
        a, b = sorted((current, other))
        current_size += sizes[other]
        rows.append([a, b, top, current_size])
        current = next_id
        next_id += 1
    return np.array(rows, dtype=float), leaves, n_forced


def girvan_newman_tree(graph: nx.Graph, weight: str | None = "weight") -> GirvanNewmanTree:
    """Levels, linkage matrix, and the most modular level, in one call."""
    levels = girvan_newman_levels(graph, weight=weight)
    linkage, leaves, n_forced = levels_to_linkage(levels)
    modularity = [nx.community.modularity(graph, level, weight=weight) for level in levels]
    return GirvanNewmanTree(
        levels=levels,
        leaves=leaves,
        linkage=linkage,
        n_forced_joins=n_forced,
        modularity=modularity,
        best_level=int(np.argmax(modularity)),
    )
