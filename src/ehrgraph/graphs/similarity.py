"""Bipartite patient-code vectors -> patient-patient similarity graph
(PROJECT_GUIDE.md Module A2, issue #5). Three stages, each its own small
function so the pipeline can be inspected/tested one step at a time:

  build_patient_vectors()  (issue #4, ehrgraph.graphs.vectorize)
      -> raw patient x code matrix
  to_tfidf()
      -> IDF-weighted, L2-normalized patient x code matrix
  cosine_similarity_matrix()
      -> patient x patient similarity matrix (dense diagonal zeroed)
  top_k_sparsify() + mutual_knn()
      -> sparse, symmetric patient x patient adjacency
  to_networkx()
      -> the actual graph object

**Why TF-IDF on top of the binary+per_patient vectors, not raw counts:**
PROJECT_GUIDE.md item 6 picked binary+per_patient as the primary
vectorization specifically to be immune to the "just measures how often
the patient was seen" utilization confound -- feeding raw per-occurrence
counts into TF-IDF here would quietly reintroduce that same confound
through the back door (TF-IDF's "TF" term rewards repeated codes). IDF
alone (how rare a code is across the whole cohort) still does real work on
top of a binary vector -- it downweights near-universal codes and
upweights rare, distinguishing ones -- without ever rewarding volume.
`to_tfidf` applies exactly that: on a 0/1 input, the "term frequency"
part is trivially 0/1, so what actually changes each value is the IDF
scaling.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity


def to_tfidf(matrix: sp.spmatrix) -> sp.csr_matrix:
    """IDF-weight and L2-normalize a patient x code matrix. Works on any
    of build_patient_vectors' weightings, but is intended for the primary
    binary+per_patient config -- see module docstring."""
    return TfidfTransformer(norm="l2").fit_transform(matrix)


def cosine_similarity_matrix(tfidf_matrix: sp.spmatrix) -> sp.csr_matrix:
    """Patient x patient cosine similarity, sparse throughout (never
    densifies the full N x N matrix). The diagonal (a patient's similarity
    to themselves) is zeroed -- a patient is never their own neighbor."""
    sim = cosine_similarity(tfidf_matrix, dense_output=False).tocsr()
    sim.setdiag(0)
    sim.eliminate_zeros()
    return sim


def top_k_sparsify(sim_matrix: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep only each patient's top-k most similar neighbors (by row).
    Operates on each row's existing nonzero entries directly (no O(n^2)
    densification), so this stays cheap even though it's a Python loop.

    NOT symmetric on its own: patient i having patient j in its top-k does
    not mean j has i in its own -- see mutual_knn/union_knn below, and
    PROJECT_GUIDE.md Module A2's correction on this exact point.
    """
    sim_matrix = sim_matrix.tocsr()
    n = sim_matrix.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(n):
        start, end = sim_matrix.indptr[i], sim_matrix.indptr[i + 1]
        row_cols = sim_matrix.indices[start:end]
        row_vals = sim_matrix.data[start:end]
        if len(row_vals) == 0:
            continue
        top_idx = np.argpartition(row_vals, -k)[-k:] if len(row_vals) > k else range(len(row_vals))
        for idx in top_idx:
            if row_vals[idx] > 0:
                rows.append(i)
                cols.append(row_cols[idx])
                vals.append(row_vals[idx])
    return sp.csr_matrix((vals, (rows, cols)), shape=sim_matrix.shape)


def mutual_knn(top_k_matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Keep an edge only if BOTH sides picked each other -- sparser,
    cleaner communities. Primary sparsification for this project."""
    return top_k_matrix.minimum(top_k_matrix.T).tocsr()


def union_knn(top_k_matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Keep an edge if EITHER side picked it -- denser. Secondary
    sensitivity check only (PROJECT_GUIDE.md "Can improve" item 5)."""
    return top_k_matrix.maximum(top_k_matrix.T).tocsr()


def to_networkx(adjacency: sp.csr_matrix, patient_ids: list[int]) -> nx.Graph:
    """Turn a symmetric sparse adjacency matrix into an undirected,
    weighted networkx graph, with every patient present as a node even if
    they end up with no edges (isolated nodes are a real finding to keep,
    not to silently drop)."""
    graph = nx.Graph()
    graph.add_nodes_from(patient_ids)
    coo = adjacency.tocoo()
    for i, j, weight in zip(coo.row, coo.col, coo.data, strict=True):
        if i < j:  # each undirected edge appears twice in a symmetric matrix
            graph.add_edge(patient_ids[i], patient_ids[j], weight=float(weight))
    return graph


def graph_summary(graph: nx.Graph) -> dict:
    """Isolated-node count and largest-connected-component size -- a real
    Project 1 finding about this population's structure, not just a
    data-cleaning footnote (PROJECT_GUIDE.md Module A2)."""
    n_nodes = graph.number_of_nodes()
    isolated = [n for n in graph.nodes if graph.degree(n) == 0]
    components = list(nx.connected_components(graph))
    largest_cc = max(components, key=len) if components else set()
    return {
        "n_nodes": n_nodes,
        "n_edges": graph.number_of_edges(),
        "n_isolated": len(isolated),
        "isolated_fraction": len(isolated) / n_nodes if n_nodes else 0.0,
        "n_components": len(components),
        "largest_cc_size": len(largest_cc),
        "largest_cc_fraction": len(largest_cc) / n_nodes if n_nodes else 0.0,
    }
