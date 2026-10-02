"""The disease-trajectory graph (issue #10, PROJECT_GUIDE.md item 15).

A different kind of graph from everything else in this project: nodes are
*diagnoses*, not patients, and an edge A -> B means "having A measurably
raises the chance of later developing B" -- a directed, explanatory graph
of disease progression, not a similarity graph used for prediction. It
feeds the Girvan-Newman dendrogram in issue #14.

**Construction rule** (previously unspecified -- see PROJECT_GUIDE.md):
  1. Nodes: the ~200 most common condition codes across the whole
     population (not one task's cohort -- disease progression is a
     population-level relationship).
  2. For each ordered pair (A, B), A != B: relative risk of B occurring
     within a fixed time window after A's first occurrence, among
     patients who have A, versus B's unconditional prevalence in the
     whole population:

         RR(A -> B) = P(B within window | has A, B after A)
                      -----------------------------------------
                              P(B) unconditionally

     A ratio > 1 means "B follows A more often than you'd expect by
     chance"; this is a simplified, tractable relative risk (comparing
     against the population baseline, not a matched unexposed group) --
     appropriate for an explanatory/descriptive graph, not a claim of
     causation.
  3. Only kept if well-supported (>= min_support patients experienced
     A-then-B) and statistically significant under a one-sided binomial
     test against the population baseline rate, Bonferroni-corrected
     across all possible ordered pairs (200*199, not just the ones that
     happened to have any support -- the stricter, standard choice).
  4. Symmetrized via max(RR_AB, RR_BA) before Girvan-Newman, which needs
     an undirected graph -- this discards directionality, a real
     tradeoff to state explicitly in the report, not hide.

Defaults (window=365 days, min_support=10) were chosen by checking the
real data first, not picked blind -- see notebooks/06_disease_trajectory_graph.ipynb.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Protocol

import networkx as nx
import pandas as pd
from scipy.stats import binomtest

from ehrgraph.data.loader import Event


class SubjectLike(Protocol):
    subject_id: int
    events: Iterable[Event]


class DatabaseLike(Protocol):
    def __getitem__(self, subject_id: int) -> SubjectLike: ...


def first_occurrence_times(
    db: DatabaseLike, subject_ids: Iterable[int], table: str, codes_of_interest: set[str]
) -> dict[int, dict[str, datetime]]:
    """For each patient, the earliest time each code-of-interest appears
    (restricted to `table`). Uses first occurrence, not every occurrence
    -- trajectory analysis asks "when did this condition start," and a
    code recorded 50 times shouldn't count as 50 separate onsets."""
    result: dict[int, dict[str, datetime]] = {}
    for sid in subject_ids:
        subject = db[sid]
        first_times: dict[str, datetime] = {}
        for event in subject.events:
            if event.table != table or event.code not in codes_of_interest:
                continue
            existing = first_times.get(event.code)
            if existing is None or event.time < existing:
                first_times[event.code] = event.time
        result[sid] = first_times
    return result


def compute_relative_risks(
    first_times: dict[int, dict[str, datetime]],
    codes: list[str],
    window: timedelta,
    min_support: int,
    n_population: int,
) -> pd.DataFrame:
    """One row per ordered (a, b) pair that has at least `min_support`
    patients going a-then-b within `window`. Columns: a, b, n_a
    (patients with a), n_a_then_b (of those, how many got b within the
    window), n_b (population-wide prevalence of b), rr, pvalue
    (one-sided binomial test: is the a-then-b rate significantly GREATER
    than b's population base rate -- not just different from it, since
    only elevated risk is the thing this graph reports)."""
    patients_with_code: dict[str, list[int]] = {code: [] for code in codes}
    n_code: dict[str, int] = dict.fromkeys(codes, 0)
    for patient_id, times in first_times.items():
        for code in times:
            patients_with_code[code].append(patient_id)
            n_code[code] += 1

    rows = []
    for a in codes:
        a_patients = patients_with_code[a]
        n_a = len(a_patients)
        if n_a == 0:
            continue
        b_counts: dict[str, int] = {}
        for patient_id in a_patients:
            a_time = first_times[patient_id][a]
            for b, b_time in first_times[patient_id].items():
                if b != a and a_time < b_time <= a_time + window:
                    b_counts[b] = b_counts.get(b, 0) + 1
        for b, n_a_then_b in b_counts.items():
            if n_a_then_b < min_support:
                continue
            n_b = n_code[b]
            baseline_rate = n_b / n_population
            rr = (n_a_then_b / n_a) / baseline_rate if baseline_rate > 0 else float("inf")
            pvalue = binomtest(n_a_then_b, n_a, baseline_rate, alternative="greater").pvalue
            rows.append((a, b, n_a, n_a_then_b, n_b, rr, pvalue))

    return pd.DataFrame(rows, columns=["a", "b", "n_a", "n_a_then_b", "n_b", "rr", "pvalue"])


def bonferroni_significant(rr_df: pd.DataFrame, n_codes: int, alpha: float = 0.05) -> pd.DataFrame:
    """Filters to pairs significant after Bonferroni-correcting `alpha`
    across every possible ordered pair among `n_codes` nodes (n*(n-1)),
    not just the pairs that happened to have enough support -- the
    stricter, standard choice, since the "family" of hypotheses under
    consideration is every pair the study design could have tested."""
    n_possible_pairs = n_codes * (n_codes - 1)
    corrected_alpha = alpha / n_possible_pairs
    return rr_df[rr_df["pvalue"] < corrected_alpha].copy()


def build_trajectory_graph(
    significant_df: pd.DataFrame, code_names: dict[str, str] | None = None
) -> nx.DiGraph:
    """One directed edge a -> b per significant row, weighted by RR.
    Nodes carry a human-readable `name` attribute when `code_names` is
    given (code -> concept name, e.g. from ehrgraph.data.vocabulary)."""
    graph = nx.DiGraph()
    for _, row in significant_df.iterrows():
        for code in (row["a"], row["b"]):
            if code not in graph:
                graph.add_node(code, name=(code_names or {}).get(code, code))
        graph.add_edge(row["a"], row["b"], weight=float(row["rr"]))
    return graph


def symmetrize_max(graph: nx.DiGraph) -> nx.Graph:
    """Undirected version for Girvan-Newman (issue #14): edge weight is
    max(RR_AB, RR_BA) where both directions exist, or whichever direction
    exists where only one does. This is a stated tradeoff, not an
    accident: it discards which direction the relationship runs."""
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        weight = data["weight"]
        if undirected.has_edge(u, v):
            weight = max(weight, undirected[u][v]["weight"])
        undirected.add_edge(u, v, weight=weight)
    return undirected


class _MappedConditions:
    """Read-only view of a meds_reader database whose condition events
    carry mapped codes (e.g. ICD -> SNOMED standard concept). Only
    condition events are kept -- all the trajectory builder reads."""

    def __init__(self, db, code_map: dict[str, str]):
        self._db, self._map = db, code_map

    def __iter__(self):
        return iter(self._db)

    def __getitem__(self, subject_id):
        from types import SimpleNamespace

        events = [
            SimpleNamespace(time=e.time, code=self._map.get(e.code, e.code), table=e.table)
            for e in self._db[subject_id].events
            if getattr(e, "table", None) == "condition"
        ]
        return SimpleNamespace(subject_id=subject_id, events=events)


def build_population_trajectory_graph(
    db,
    concept_table: pd.DataFrame | None = None,
    n_codes: int = 200,
    window: timedelta = timedelta(days=365),
    min_support: int = 10,
    alpha: float = 0.05,
    code_map: dict[str, str] | None = None,
) -> tuple[nx.DiGraph, dict[str, str]]:
    """Notebook 06's full construction in one call, so notebook 08 (the
    Girvan-Newman dendrogram) builds exactly the same graph: the `n_codes`
    most common condition codes across every patient, relative risks within
    `window`, `min_support`, Bonferroni over all n*(n-1) ordered pairs.
    Returns the directed graph and code -> readable name.

    `code_map` (e.g. from `vocabulary.standard_concept_map`) rewrites codes
    before anything is counted, so an ICD-9 and an ICD-10 code for the same
    disease become one node instead of two era-specific ones."""
    from ehrgraph.data.cohort import patient_code_counts
    from ehrgraph.data.vocabulary import resolve_codes

    if code_map is not None:
        db = _MappedConditions(db, code_map)
    patient_ids = list(db)
    counts = patient_code_counts(db, patient_ids, table="condition")
    top_codes = [code for code, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n_codes]]

    code_names = {c: c for c in top_codes}
    if concept_table is not None:
        resolved = resolve_codes(pd.Series(top_codes), concept_table).set_index("code")
        code_names |= resolved["concept_name"].dropna().to_dict()

    first_times = first_occurrence_times(
        db, patient_ids, table="condition", codes_of_interest=set(top_codes)
    )
    rr_df = compute_relative_risks(
        first_times,
        codes=top_codes,
        window=window,
        min_support=min_support,
        n_population=len(patient_ids),
    )
    significant = bonferroni_significant(rr_df, n_codes=len(top_codes), alpha=alpha)
    return build_trajectory_graph(significant, code_names=code_names), code_names
