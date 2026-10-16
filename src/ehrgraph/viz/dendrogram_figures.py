"""Dendrogram figures for the disease-trajectory graph (issue #14).

Diagnosis names and codes only -- nothing patient-level -- so these PNGs are
safe for results/aggregate/figures/.

Community colors follow the categorical palette in fixed order, largest
community first; past seven communities the rest are drawn in neutral gray
rather than cycling hues (a recycled hue would claim two unrelated groups
belong together).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram

from ehrgraph.netstats.dendrogram import GirvanNewmanTree
from ehrgraph.viz.network_figures import INK_2, INK_3

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
OTHER = "#b5b4ad"


def _leaf_sets(tree: GirvanNewmanTree) -> dict[int, set]:
    n = len(tree.leaves)
    members = {i: {i} for i in range(n)}
    for t, (a, b, _, _) in enumerate(tree.linkage):
        members[n + t] = members[int(a)] | members[int(b)]
    return members


def community_colors(tree: GirvanNewmanTree, level: int | None = None) -> dict:
    """leaf -> color for the communities at `level` (default: most modular)."""
    level = tree.best_level if level is None else level
    ranked = sorted(tree.levels[level], key=len, reverse=True)
    colors = {}
    for i, community in enumerate(ranked):
        color = CATEGORICAL[i] if i < len(CATEGORICAL) and len(community) > 1 else OTHER
        for node in community:
            colors[node] = color
    return colors


def plot_dendrogram(
    tree: GirvanNewmanTree,
    leaf_color: dict,
    labels: dict | None = None,
    ax=None,
    orientation: str = "left",
    label_width: int = 42,
):
    """Draws the Girvan-Newman tree. Links inside one color group take
    that color; links joining groups are gray. Joins between components
    that never shared an edge (`tree.n_forced_joins`) sit in a shaded band
    at the top -- the algorithm has no evidence for how those pieces relate."""
    n = len(tree.leaves)
    members = _leaf_sets(tree)
    leaf_colors = [leaf_color.get(leaf, OTHER) for leaf in tree.leaves]

    def link_color(k):
        colors = {leaf_colors[i] for i in members[k]}
        return colors.pop() if len(colors) == 1 and OTHER not in colors else INK_3

    if ax is None:
        _, ax = plt.subplots()
    text = (
        [(labels or {}).get(leaf, str(leaf))[:label_width] for leaf in tree.leaves]
        if labels is not None
        else None
    )
    result = dendrogram(
        tree.linkage,
        orientation=orientation,
        labels=text,
        no_labels=labels is None,
        link_color_func=link_color,
        leaf_font_size=8,
        ax=ax,
    )
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    if tree.n_forced_joins:
        real_top = tree.linkage[: n - 1 - tree.n_forced_joins, 2].max()
        top = tree.linkage[:, 2].max()
        span = (real_top + 0.5, top + 0.5)
        (ax.axvspan if orientation in ("left", "right") else ax.axhspan)(
            *span, color="#efeee9", zorder=0
        )
        if orientation in ("left", "right"):
            # inside the band, rotated -- clear of the title and leaf labels
            ax.text(
                np.mean(span),
                0.5,
                "pieces with no shared links",
                rotation=90,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="center",
                color=INK_2,
                fontsize=8,
            )
        else:
            ax.text(
                1.0,
                np.mean(span),
                " pieces with\n no shared links",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                color=INK_2,
                fontsize=8,
            )

    if labels is not None:
        tick_labels = (
            ax.get_yticklabels() if orientation in ("left", "right") else ax.get_xticklabels()
        )
        by_text = dict(zip(text, leaf_colors, strict=True))
        for tick in tick_labels:
            tick.set_color(INK_2)
            color = by_text.get(tick.get_text())
            if color and color != OTHER:
                tick.set_fontweight("bold")
    axis_label = "split order (higher = split earlier)"
    if orientation in ("left", "right"):
        ax.set_xlabel(axis_label, color=INK_2)
    else:
        ax.set_ylabel(axis_label, color=INK_2)
    return result


def community_legend(
    tree: GirvanNewmanTree, names: dict, graph, level: int | None = None, k: int = 7
):
    """Legend handles naming each colored community by its best-connected
    member (highest degree in `graph`), so readers see "heart failure
    group", not "community 3"."""
    level = tree.best_level if level is None else level
    ranked = [c for c in sorted(tree.levels[level], key=len, reverse=True) if len(c) > 1][:k]
    handles = []
    for i, community in enumerate(ranked):
        example = max(sorted(community, key=str), key=graph.degree)
        handles.append(
            Patch(
                color=CATEGORICAL[i], label=f"{len(community)}: {names.get(example, example)[:34]}…"
            )
        )
    return handles
