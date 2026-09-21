import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from ehrgraph.graphs.similarity import (
    cosine_similarity_matrix,
    graph_summary,
    mutual_knn,
    to_networkx,
    to_tfidf,
    top_k_sparsify,
    union_knn,
)


def test_to_tfidf_downweights_common_codes():
    # 4 patients x 2 codes: code 0 is universal (every patient has it),
    # code 1 is rare (only patient 0 has it). IDF should make code 1's
    # weight larger than code 0's for patient 0, even though both are 1
    # in the raw binary input.
    matrix = sp.csr_matrix(
        np.array(
            [
                [1, 1],
                [1, 0],
                [1, 0],
                [1, 0],
            ],
            dtype=float,
        )
    )
    tfidf = to_tfidf(matrix).toarray()
    assert tfidf[0, 1] > tfidf[0, 0]


def test_cosine_similarity_identical_patients_score_one():
    matrix = sp.csr_matrix(np.array([[1, 1, 0], [1, 1, 0]], dtype=float))
    tfidf = to_tfidf(matrix)
    sim = cosine_similarity_matrix(tfidf).toarray()
    assert sim[0, 1] == pytest.approx(1.0)


def test_cosine_similarity_disjoint_patients_score_zero():
    matrix = sp.csr_matrix(np.array([[1, 0], [0, 1]], dtype=float))
    tfidf = to_tfidf(matrix)
    sim = cosine_similarity_matrix(tfidf).toarray()
    assert sim[0, 1] == 0.0


def test_cosine_similarity_diagonal_is_zeroed():
    matrix = sp.csr_matrix(np.array([[1, 1], [1, 0], [0, 1]], dtype=float))
    tfidf = to_tfidf(matrix)
    sim = cosine_similarity_matrix(tfidf).toarray()
    assert np.allclose(np.diag(sim), 0.0)


def test_top_k_sparsify_keeps_only_k_per_row():
    # patient 0 is similar to 1, 2, and 3 with increasing strength;
    # k=1 should keep only the strongest (patient 3).
    sim = sp.csr_matrix(
        np.array(
            [
                [0.0, 0.1, 0.2, 0.9],
                [0.1, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0, 0.0],
            ]
        )
    )
    sparse = top_k_sparsify(sim, k=1)
    row0 = sparse.getrow(0).toarray().flatten()
    assert (row0 > 0).sum() == 1
    assert row0[3] == 0.9


def test_top_k_sparsify_is_not_symmetric():
    # patient 0's only neighbor is 1 (weak), but patient 1's top choice is
    # patient 2 (strong) -- patient 0 should not automatically appear in
    # patient 1's kept row.
    sim = sp.csr_matrix(
        np.array(
            [
                [0.0, 0.1, 0.0],
                [0.1, 0.0, 0.9],
                [0.0, 0.9, 0.0],
            ]
        )
    )
    sparse = top_k_sparsify(sim, k=1)
    assert sparse[0, 1] == 0.1  # 0 picked 1
    assert sparse[1, 0] == 0.0  # but 1 did not pick 0 back (picked 2 instead)


def test_mutual_knn_keeps_only_reciprocated_edges():
    # An already-asymmetric top-k result, like top_k_sparsify would produce:
    # patient 0 picked patient 1 (weight 0.1), but patient 1 did NOT pick
    # patient 0 back -- patient 1's own top-k pick was patient 2 (0.9),
    # which patient 2 reciprocated.
    top_k = sp.csr_matrix(
        np.array(
            [
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.9],
                [0.0, 0.9, 0.0],
            ]
        )
    )
    mutual = mutual_knn(top_k).toarray()
    assert mutual[0, 1] == 0.0  # not reciprocated
    assert mutual[1, 2] == pytest.approx(0.9)  # reciprocated


def test_union_knn_keeps_either_direction():
    top_k = sp.csr_matrix(
        np.array(
            [
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.9],
                [0.0, 0.9, 0.0],
            ]
        )
    )
    union = union_knn(top_k).toarray()
    assert union[0, 1] == pytest.approx(0.1)  # kept even though one-sided
    assert union[1, 0] == pytest.approx(0.1)  # symmetric


def test_to_networkx_builds_correct_edges_and_weights():
    adjacency = sp.csr_matrix(
        np.array(
            [
                [0.0, 0.5, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )
    graph = to_networkx(adjacency, patient_ids=[10, 20, 30])
    assert set(graph.nodes) == {10, 20, 30}
    assert graph.has_edge(10, 20)
    assert graph[10][20]["weight"] == pytest.approx(0.5)
    assert graph.degree[30] == 0  # isolated node still present


def test_graph_summary_counts_isolated_and_largest_component():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2, 3, 4, 5])
    graph.add_edge(1, 2, weight=0.5)
    graph.add_edge(2, 3, weight=0.5)
    # 4 and 5 are isolated

    summary = graph_summary(graph)
    assert summary["n_nodes"] == 5
    assert summary["n_isolated"] == 2
    assert summary["isolated_fraction"] == pytest.approx(0.4)
    assert summary["largest_cc_size"] == 3
    assert summary["largest_cc_fraction"] == pytest.approx(0.6)
    assert summary["n_components"] == 3  # {1,2,3}, {4}, {5}
