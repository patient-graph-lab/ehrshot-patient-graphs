"""Turn each patient's medical-code history into a numeric vector --
Module A2's vectorization stage (PROJECT_GUIDE.md item 6, issue #4). This
produces the raw patient x code matrix that issue #5's TF-IDF + cosine
similarity step consumes; it does NOT apply IDF weighting itself -- that's
a graph-construction decision, not a vectorization one.

Two independent axes to sweep (PROJECT_GUIDE.md item 6 -- this is now an
experiment axis, not a locked pre-decision):

  dedup     what counts as "one occurrence" before counting:
              "per_occurrence" -- every event row counts, no collapsing.
              "per_visit"      -- collapse repeats of the same code within
                                  the same visit to one.
              "per_patient"    -- collapse to presence/absence: did this
                                  patient ever have this code, regardless
                                  of how many times or in how many visits.

  weighting how the deduped counts become a vector value:
              "binary"           -- 1 if the (deduped) count is > 0, else
                                     absent. Note: binary is dedup-INVARIANT
                                     by construction -- presence/absence
                                     doesn't care how many times or where a
                                     code occurred, so binary+per_occurrence,
                                     binary+per_visit, and binary+per_patient
                                     all produce the identical matrix. dedup
                                     only changes the result for "count" and
                                     "count_normalized".
              "count"            -- the raw deduped count.
              "count_normalized" -- deduped count / that patient's total
                                     code count for this table (a relative-
                                     frequency row, like TF without IDF).

Primary reported configuration (PROJECT_GUIDE.md item 6): binary +
per_patient -- presence/absence only is *structurally immune* to the
"isn't this just measuring how often the patient was seen" utilization
confound, since counts never enter the vector at all.

Call this once per OMOP source table ("condition", "drug_exposure",
"procedure_occurrence" -- PROJECT_GUIDE.md's "per relation type" step),
not once with all tables mixed together, since the three relation types
stay separate until Project 1's multi-relational reporting collapses them.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import scipy.sparse as sp

from ehrgraph.data.loader import Event

WEIGHTINGS = ("binary", "count", "count_normalized")
DEDUPS = ("per_occurrence", "per_visit", "per_patient")


class SubjectLike(Protocol):
    subject_id: int
    events: Iterable[Event]


class DatabaseLike(Protocol):
    def __getitem__(self, subject_id: int) -> SubjectLike: ...


def _deduped_counts(
    events: Iterable[Event], dedup: str, rollup: dict[str, str] | None
) -> dict[str, int]:
    """One patient's {code: count} after applying the ontology rollup (if
    any) and the chosen dedup granularity. Rollup is applied per-event,
    before counting, so two different codes that roll up to the same
    ancestor are counted together -- not counted separately and then
    merged, which would double-count a single visit that happened to use
    two near-duplicate codes."""
    if dedup not in DEDUPS:
        raise ValueError(f"Unknown dedup {dedup!r}, expected one of {DEDUPS}")

    counts: dict[str, int] = {}
    seen: set[tuple] = set()
    for event in events:
        code = rollup.get(event.code, event.code) if rollup else event.code
        if dedup == "per_occurrence":
            dedup_key = None  # every row counts; no collapsing
        elif dedup == "per_visit":
            dedup_key = (code, event.visit_id)
        else:  # per_patient
            dedup_key = (code,)
        if dedup_key is not None:
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
        counts[code] = counts.get(code, 0) + 1
    return counts


def build_patient_vectors(
    db: DatabaseLike,
    subject_ids: Iterable[int],
    table: str,
    weighting: str = "binary",
    dedup: str = "per_patient",
    rollup: dict[str, str] | None = None,
) -> tuple[sp.csr_matrix, list[int], list[str]]:
    """One row per subject, one column per distinct code seen (restricted
    to `table`). Returns (matrix, patient_id_order, code_order): matrix[i,
    j] is patient_id_order[i]'s value for code_order[j].

    `rollup` is an optional {code: ancestor_code} map (from OMOP's
    concept_ancestor, per Module A2's ontology-rollup robustness check) --
    building that map from real data is a separate concern; this function
    just applies one if given.

    Codes are sorted alphabetically so the same inputs always produce the
    same column order -- matters for reproducibility (issue #9's
    determinism test).
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"Unknown weighting {weighting!r}, expected one of {WEIGHTINGS}")

    subject_ids = list(subject_ids)
    per_patient_counts: dict[int, dict[str, int]] = {}
    all_codes: set[str] = set()
    for sid in subject_ids:
        events = (e for e in db[sid].events if e.table == table)
        counts = _deduped_counts(events, dedup, rollup)
        per_patient_counts[sid] = counts
        all_codes.update(counts)

    code_order = sorted(all_codes)
    code_index = {code: j for j, code in enumerate(code_order)}

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i, sid in enumerate(subject_ids):
        counts = per_patient_counts[sid]
        total = sum(counts.values())
        for code, count in counts.items():
            if weighting == "binary":
                value = 1.0
            elif weighting == "count":
                value = float(count)
            else:  # count_normalized
                value = count / total if total else 0.0
            rows.append(i)
            cols.append(code_index[code])
            vals.append(value)

    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(len(subject_ids), len(code_order)))
    return matrix, subject_ids, code_order
