from datetime import datetime

import networkx as nx

from ehrgraph.data.loader import events_all, events_before
from ehrgraph.graphs.null_model import build_null_graph
from ehrgraph.graphs.similarity import (
    cosine_similarity_matrix,
    mutual_knn,
    to_networkx,
    to_tfidf,
    top_k_sparsify,
)
from ehrgraph.graphs.vectorize import build_node_vectors
from tests.fixtures.make_synthetic import cohort_for_vectorization, subject_with_events_around


def test_events_before_excludes_boundary_and_future():
    prediction_time = datetime(2020, 12, 1)
    subject = subject_with_events_around(prediction_time)

    result = events_before(subject, prediction_time)

    codes = {e.code for e in result}
    assert codes == {"before/1", "before/2"}
    assert all(e.time < prediction_time for e in result)


def test_events_before_is_not_a_no_op():
    """The failure mode that matters most: censoring silently doing nothing
    would still produce a graph, just a wrong one, with no error anywhere."""
    prediction_time = datetime(2020, 12, 1)
    subject = subject_with_events_around(prediction_time)

    safe = events_before(subject, prediction_time)
    leaky = events_all(subject)

    assert len(safe) < len(leaky)
    assert {e.code for e in safe} != {e.code for e in leaky}


def test_events_before_property_no_future_event_ever_included():
    """Property-style check across several prediction times on the same subject."""
    subject = subject_with_events_around(datetime(2020, 12, 1))
    for cutoff in [datetime(2019, 1, 1), datetime(2020, 6, 1), datetime(2021, 6, 1)]:
        result = events_before(subject, cutoff)
        assert all(e.time < cutoff for e in result)


def _build_graph(nodes, db, censor: bool, seed: int) -> nx.Graph:
    """The real safe/leaky graph-construction pipeline (issues #4-#8), run
    end to end on a synthetic cohort -- used here purely to check
    determinism, not correctness (that's covered elsewhere per-stage)."""
    matrix, node_order, _ = build_node_vectors(
        db, nodes, table="condition", censor=censor, weighting="binary", dedup="per_patient"
    )
    sim = cosine_similarity_matrix(to_tfidf(matrix))
    graph = to_networkx(mutual_knn(top_k_sparsify(sim, k=2)), node_order)
    return build_null_graph(graph, seed=seed)


def test_determinism_same_seed_gives_identical_graph_hash():
    """The last of issue #9's four tests, closed out now that issues #6-#8
    (the real safe/leaky/null graph pipeline) exist to hash. Same inputs
    and same seed must produce byte-for-byte the same graph -- if this
    ever flakes, something in the pipeline picked up hidden randomness
    (unordered set/dict iteration, an unseeded RNG call, etc.), and every
    result downstream would be unreproducible."""
    db = cohort_for_vectorization()
    today = datetime(2024, 1, 1)
    nodes = [(10, today), (11, today), (12, today)]

    graph_a = _build_graph(nodes, db, censor=True, seed=7)
    graph_b = _build_graph(nodes, db, censor=True, seed=7)

    hash_a = nx.weisfeiler_lehman_graph_hash(graph_a, edge_attr="weight")
    hash_b = nx.weisfeiler_lehman_graph_hash(graph_b, edge_attr="weight")
    assert hash_a == hash_b
