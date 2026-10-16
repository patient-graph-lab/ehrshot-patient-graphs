"""Figures for one task's network analysis (notebooks 07 and 10).

Lifted out of notebook 07 so Task A and Task B draw identical figures and
the cross-task comparison (#17) compares like with like. Every figure is
built from aggregate summaries or anonymous dots -- no patient IDs, no
labels on individual nodes -- so the PNGs are safe for results/aggregate/.

Colors: the first three slots of a colorblind-validated categorical palette
(safe / leaky / null), fixed per condition across every figure.
"""

from __future__ import annotations

import random
from pathlib import Path

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ehrgraph.netstats.centrality import degree_distribution
from ehrgraph.netstats.task_analysis import TaskAnalysis, TaskGraphs

COLORS = {"safe": "#2a78d6", "leaky": "#eb6834", "null": "#1baf7a"}
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
POS, NEG = "#4a3aa7", "#d4d3cc"
CONDITIONS = ("safe", "leaky", "null")
#: what the notebooks and report call each graph
LABELS = {"safe": "honest", "leaky": "cheating", "null": "random"}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "axes.edgecolor": INK_3,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_3,
            "ytick.color": INK_3,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "font.size": 11,
            "legend.fontsize": 9,
        }
    )


def save(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def degree_figure(tg: TaskGraphs, k: int = 10):
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.4
    for offset, name in ((-width / 2, "safe"), (width / 2, "leaky")):
        counts = degree_distribution(tg.graphs[name]).value_counts()
        counts = counts.reindex(range(k + 1), fill_value=0)
        ax.bar(
            counts.index + offset,
            counts / counts.sum(),
            width=width - 0.04,
            color=COLORS[name],
            label=name,
        )
    ax.set_xlabel(f"degree (number of links; mutual k-NN caps it at {k})")
    ax.set_ylabel("share of all nodes")
    ax.set_title("Degree distribution (0 = isolated; null = safe by construction)")
    ax.set_xticks(range(k + 1))
    ax.grid(axis="x", visible=False)
    ax.legend()
    fig.tight_layout()
    return fig


def centrality_figure(an: TaskAnalysis):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, measure in zip(axes, ("betweenness", "closeness", "eigenvector"), strict=True):
        for name in CONDITIONS:
            values = np.sort(an.centrality[name][measure].dropna().to_numpy())
            ax.plot(
                values,
                np.linspace(0, 1, len(values)),
                color=COLORS[name],
                lw=2,
                ls="--" if name == "null" else "-",
                label=name,
            )
        ax.set_title(measure)
        ax.set_xlabel(f"{measure} centrality")
        if measure in ("betweenness", "eigenvector"):
            ax.set_xscale("symlog", linthresh=1e-4)
    axes[0].set_ylabel("share of nodes ≤ value")
    axes[0].legend(loc="lower right")
    fig.suptitle("Centrality distributions (cumulative)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def community_outcome_figure(tg: TaskGraphs, an: TaskAnalysis, outcome: str, min_size: int = 20):
    """Outcome rate in each community (>= min_size stays), sorted low to high."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for name in CONDITIONS:
        o = an.community_outcome[name]
        rates = o["rates"]["mean"].sort_values().to_numpy()
        ax.plot(
            np.linspace(0, 1, len(rates)),
            rates,
            marker="o",
            ms=4,
            lw=1.5,
            color=COLORS[name],
            ls="--" if name == "null" else "-",
            label=f"{LABELS[name]} ({len(rates)} communities)",
        )
    ax.axhline(tg.prevalence, color=INK_3, lw=1, ls=":")
    ax.text(
        1.0,
        tg.prevalence + 0.02,
        f"overall {tg.prevalence:.0%}",
        color=INK_3,
        fontsize=9,
        ha="right",
    )
    ax.set_xlabel("communities, from lowest to highest rate")
    ax.set_xticks([])
    ax.set_ylabel(f"share with {outcome}")
    ax.set_ylim(0, 1)
    ax.set_title(f"{outcome.capitalize()}: rate in each community")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def _layout_lcc(graph: nx.Graph):
    lcc = graph.subgraph(max(nx.connected_components(graph), key=len))
    order = list(lcc.nodes)
    index = {n: i for i, n in enumerate(order)}
    ig_graph = ig.Graph(n=len(order), edges=[(index[u], index[v]) for u, v in lcc.edges()])
    coords = np.array(ig_graph.layout_fruchterman_reingold(niter=1500).coords)
    return lcc, order, index, coords


def layout_figure(tg: TaskGraphs, positive: str, negative: str, seed: int = 0):
    random.seed(seed)  # igraph's layout draws from Python's RNG once pointed at it
    ig.set_random_number_generator(random)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9))
    for ax, name in zip(axes, CONDITIONS, strict=True):
        lcc, order, index, coords = _layout_lcc(tg.graphs[name])
        segs = np.array([[coords[index[u]], coords[index[v]]] for u, v in lcc.edges()])
        ax.add_collection(LineCollection(segs, colors=INK_3, linewidths=0.1, alpha=0.08))
        y = np.array([tg.labels[n] for n in order])
        ax.scatter(*coords[y == 0].T, s=2, c=NEG, linewidths=0)
        ax.scatter(*coords[y == 1].T, s=3, c=POS, linewidths=0)
        ax.set_title(f"{LABELS[name]} ({len(order):,} stays)")
        ax.set_aspect("equal")
        ax.axis("off")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=POS, label=positive),
        plt.Line2D([], [], marker="o", ls="", color=NEG, label=negative),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def homophily_energy_figure(an: TaskAnalysis):
    """Homophily, plus Dirichlet energy when CLMBR fingerprints were
    available (full release); one panel otherwise."""
    panels = [("class_insensitive_homophily", "Class-insensitive homophily\n(0 = random)")]
    if "dirichlet_energy_clmbr" in an.null_homophily[0]:
        panels.append(
            ("dirichlet_energy_clmbr", "Dirichlet energy, CLMBR fingerprints\n(lower = smoother)")
        )
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 3.8), squeeze=False)
    axes = axes[0]
    ypos = {"safe": 2, "leaky": 1, "null": 0}
    for ax, (metric, title) in zip(axes, panels, strict=True):
        null_vals = np.array([r[metric] for r in an.null_homophily])
        ax.hlines(0, null_vals.min(), null_vals.max(), color=COLORS["null"], lw=6, alpha=0.35)
        ax.plot(null_vals, np.zeros_like(null_vals), "o", color=COLORS["null"], ms=5, alpha=0.7)
        for name in ("safe", "leaky"):
            value = an.homophily[name][metric]
            ax.plot(value, ypos[name], "o", color=COLORS[name], ms=10)
            ax.annotate(
                f"{value:.3f}",
                (value, ypos[name]),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                color=INK_2,
            )
        ax.annotate(
            f"{null_vals.mean():.3f} ({len(null_vals)} seeds)",
            (null_vals.max(), 0),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK_2,
        )
        ax.set_yticks([2, 1, 0], [LABELS[c] for c in CONDITIONS])
        ax.set_ylim(-0.6, 2.6)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        values = [*null_vals, *(an.homophily[n][metric] for n in ("safe", "leaky"))]
        pad = (max(values) - min(values)) * 0.25
        ax.set_xlim(min(values) - pad, max(values) + pad * 1.6)
    fig.tight_layout()
    return fig


def leakage_edges_figure(tg: TaskGraphs, an: TaskAnalysis):
    """Issue #15: (left) do the links leakage adds join same-outcome nodes
    more often than the links it removes? (right) how much of each graph's
    homophily is a patient linked to their own other admissions?"""
    fig, (ax_e, ax_h) = plt.subplots(
        1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.1]}
    )

    sets = (
        ("shared_edges", "in both", INK_3),
        ("safe_only_edges", "honest only", COLORS["safe"]),
        ("leaky_only_edges", "cheating only", COLORS["leaky"]),
    )
    rates = [an.edge_sets[key]["same_outcome_rate"] for key, _, _ in sets]
    bars = ax_e.bar(range(3), rates, color=[c for _, _, c in sets], width=0.6)
    for bar, (key, _, _) in zip(bars, sets, strict=True):
        es = an.edge_sets[key]
        ax_e.annotate(
            f"{bar.get_height():.2f}\n{es['n_edges']:,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=INK_2,
            fontsize=9,
        )
    p = tg.prevalence
    chance = p**2 + (1 - p) ** 2
    ax_e.axhline(chance, color=INK_3, lw=1, ls=":")
    ax_e.text(2.45, chance, f" chance\n {chance:.2f}", va="center", color=INK_3, fontsize=9)
    ax_e.set_xticks(range(3), [label for _, label, _ in sets])
    ax_e.set_ylim(0, 1.08)
    ax_e.set_ylabel("share with the same outcome")
    ax_e.set_title("Connections joining same-outcome stays")
    ax_e.grid(axis="x", visible=False)

    ypos = {"safe": 2, "leaky": 1, "null": 0}
    for name in CONDITIONS:
        if name == "null":
            vals = an.null_homophily
            with_sp = np.mean([r["class_insensitive_homophily"] for r in vals])
            without = np.nanmean([r["class_insensitive_homophily_excl_same_patient"] for r in vals])
        else:
            with_sp = an.homophily[name]["class_insensitive_homophily"]
            without = an.homophily[name]["class_insensitive_homophily_excl_same_patient"]
        y = ypos[name]
        ax_h.plot([with_sp, without], [y, y], color=COLORS[name], lw=1.5, alpha=0.6)
        ax_h.plot(with_sp, y, "o", color=COLORS[name], ms=9)
        ax_h.plot(without, y, "o", mfc="#fcfcfb", mec=COLORS[name], mew=2, ms=9)
        ax_h.annotate(
            f"{with_sp:.2f} → {without:.2f}",
            (max(with_sp, without), y),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            color=INK_2,
        )
    ax_h.set_yticks([2, 1, 0], [LABELS[c] for c in CONDITIONS])
    ax_h.set_ylim(-0.6, 2.6)
    ax_h.grid(axis="y", visible=False)
    ax_h.set_xlabel("homophily (0 = chance)")
    ax_h.set_title("All connections (●) vs. different patients (○)")
    lo, hi = ax_h.get_xlim()
    ax_h.set_xlim(lo, hi + (hi - lo) * 0.35)
    fig.tight_layout()
    return fig


def sir_figure(curves_by_task: dict):
    """curves_by_task: {task title: {"safe"|"leaky"|"null": curve array}}."""
    fig, axes = plt.subplots(
        1, len(curves_by_task), figsize=(2.8 * len(curves_by_task), 3.0), squeeze=False, sharey=True
    )
    for ax, (title, curves) in zip(axes[0], curves_by_task.items(), strict=True):
        for name in CONDITIONS:
            ax.plot(
                curves[name],
                color=COLORS[name],
                lw=2.2,
                ls="--" if name == "null" else "-",
                label=LABELS[name],
            )
        ax.set_title(title)
        ax.set_xlabel("time step")
    axes[0][0].set_ylabel("share of stays reached")
    axes[0][0].legend(loc="lower right")
    fig.tight_layout()
    return fig


def cascade_figure(reach_by_task: dict):
    """reach_by_task: {task title: {"safe"|"leaky"|"null": cascade_reach dict}}."""
    fig, axes = plt.subplots(
        1, len(reach_by_task), figsize=(2.8 * len(reach_by_task), 3.0), squeeze=False, sharey=True
    )
    for ax, (title, reach) in zip(axes[0], reach_by_task.items(), strict=True):
        shares = [reach[n]["cascade_positive_share"] for n in CONDITIONS]
        bars = ax.bar(
            [LABELS[n] for n in CONDITIONS],
            shares,
            color=[COLORS[n] for n in CONDITIONS],
            width=0.6,
        )
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.0%}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=INK_2,
                fontsize=9,
            )
        base = reach["safe"]["base_rate"]
        ax.axhline(base, color=INK_3, ls=":", lw=1)
        ax.text(-0.45, base, f"base {base:.0%}", va="bottom", color=INK_3, fontsize=9)
        ax.set_ylim(0, 0.5)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("reached stays with the outcome")
    fig.tight_layout()
    return fig


def era_by_year_figure(counts):
    """counts: DataFrame with columns year, ICD9CM, ICD10CM (diagnosis rows per year)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(counts["year"], counts["ICD9CM"], color=COLORS["safe"], label="ICD-9 codes", width=0.8)
    ax.bar(
        counts["year"],
        counts["ICD10CM"],
        bottom=counts["ICD9CM"],
        color=COLORS["leaky"],
        label="ICD-10 codes",
        width=0.8,
    )
    ax.axvline(2015.75, color=INK_3, ls=":", lw=1)
    ax.text(2015.85, ax.get_ylim()[1] * 0.92, "Oct 2015 switch", color=INK_2, fontsize=9)
    ax.set_ylabel("records")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax.set_title("Diagnosis records per year")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def cross_task_figure(values: dict):
    """values: {metric title: {task title: (safe, leaky, null)}} -> one dot panel per metric."""
    fig, axes = plt.subplots(1, len(values), figsize=(3.4 * len(values), 3.2), squeeze=False)
    for ax, (metric, by_task) in zip(axes[0], values.items(), strict=True):
        for x, triple in enumerate(by_task.values()):
            for dx, name, v in zip((-0.22, 0.0, 0.22), CONDITIONS, triple, strict=True):
                if v != v:  # NaN: not measured for this graph -- leave it out, don't fake a zero
                    continue
                ax.plot(
                    x + dx, v, "o", ms=9, color=COLORS[name], label=LABELS[name] if x == 0 else None
                )
                ax.annotate(
                    f"{v:.2f}",
                    (x + dx, v),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=INK_2,
                )
        ax.set_xticks(range(len(by_task)), list(by_task))
        ax.set_xlim(-0.5, len(by_task) - 0.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(metric, fontsize=11)
    axes[0][0].legend(loc="best")
    fig.tight_layout()
    return fig
