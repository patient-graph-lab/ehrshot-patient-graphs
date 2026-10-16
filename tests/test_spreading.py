import os
import subprocess
import sys

import networkx as nx
import numpy as np

from ehrgraph.netstats.spreading import cascade_reach, sir_curves, spectral_vs_leiden


def test_sir_spreads_and_never_decreases():
    graph = nx.connected_watts_strogatz_graph(200, 6, 0.1, seed=0)
    curve = sir_curves(graph, beta=0.5, gamma=0.1, n_seeds=2, steps=30, runs=5)
    assert curve[0] == 2 / 200
    assert np.all(np.diff(curve) >= 0)  # "ever infected" can only grow
    assert curve[-1] > 0.5


def test_sir_with_zero_beta_stays_at_seeds():
    graph = nx.path_graph(50)
    curve = sir_curves(graph, beta=0.0, n_seeds=3, steps=10, runs=3)
    assert np.allclose(curve, 3 / 50)


def _two_groups():
    # group A (outcome 1) and group B (outcome 0), dense inside, one bridge
    graph = nx.Graph()
    a, b = [("a", i) for i in range(20)], [("b", i) for i in range(20)]
    for group in (a, b):
        graph.add_edges_from((u, v) for i, u in enumerate(group) for v in group[i + 1 :])
    graph.add_edge(a[0], b[0])
    nx.set_edge_attributes(graph, 1.0, "weight")
    labels = {n: 1 for n in a} | {n: 0 for n in b}
    return graph, labels


def test_cascade_from_positives_reaches_mostly_positives_on_a_homophilous_graph():
    graph, labels = _two_groups()
    out = cascade_reach(graph, labels, p=0.3, seed_share=0.1, runs=20)
    assert out["base_rate"] == 0.5
    # well above the 50% base rate (the single bridge lets some of it leak out)
    assert out["cascade_positive_share"] > 0.7


def test_spectral_and_leiden_recover_two_obvious_groups():
    graph, labels = _two_groups()
    out = spectral_vs_leiden(graph, labels, k=2)
    assert out["spectral_vs_leiden_ari"] > 0.9
    assert out["spectral_eta_squared"] > 0.9
    assert out["spectral_modularity"] > 0.4


def test_sir_is_the_same_whatever_the_hash_seed():
    # Regression: starting nodes used to depend on set order (PYTHONHASHSEED).
    code = (
        "import networkx as nx\n"
        "from ehrgraph.netstats.spreading import sir_curves\n"
        "g = nx.connected_watts_strogatz_graph(300, 6, 0.05, seed=1)\n"
        "g = nx.relabel_nodes(g, lambda i: f'stay-{i}')\n"
        "print(sir_curves(g, beta=0.3, steps=10, runs=5)[10])\n"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": str(h)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for h in (1, 2, 3)
    }
    assert len(outputs) == 1, outputs
