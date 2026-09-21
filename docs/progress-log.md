# Progress Log

One entry per closed issue, written right after that issue's PR is opened. Purpose: a running, low-overhead record of what was actually implemented and why — separate from the PR diffs themselves, readable without digging through git history, and useful when writing the Project 1/2 reports later.

Each entry is written by a lightweight model pass (not the main working session) to keep this cheap to maintain. Format per entry:

```
## #<issue number> — <issue title>
**Date:** YYYY-MM-DD · **PR:** <link> · **Branch:** <name>

**What happened:** 2-4 sentences, plain language, what was built/fixed and why.

**Code and packages touched:**
- `path/to/file.py` — one line on what it does
- New dependency: `package-name` — why it was needed (omit if none)

**Verification:** how it was checked (tests run, real-data check pending in Colab, etc.)

**Follow-ups / open questions:** anything left for a later issue (omit if none)
```

---

## #1 — Verify meds_reader.SubjectDatabase against our local EHRSHOT copy
**Date:** 2026-09-20 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/21 · **Branch:** issue-1-meds-reader-wrapper

**What happened:** Built `src/ehrgraph/data/loader.py`, a thin wrapper around the meds_reader library. Verified the correct API is `meds_reader.SubjectDatabase` (not `PatientDatabase`, which was the original incorrect assumption in the project docs) against the library's real README. Implemented two separate functions—`events_before(subject, prediction_time)` for the leakage-safe condition and `events_all(subject)` for the intentionally-leaky condition—deliberately kept as two distinctly-named functions rather than one function with a boolean flag, so the safe and leaky code paths are visibly different in a code review. `open_database(path)` wraps `meds_reader.SubjectDatabase`, with the `meds_reader` import done lazily inside the function so the rest of the package doesn't have a hard dependency on that compiled library.

Also built a synthetic test fixture (`tests/fixtures/make_synthetic.py`, a fake subject with events before/at/after a prediction_time boundary, no real patient data) and three real tests in `tests/test_censoring.py`: one checking the censoring boundary is correct, one specifically checking that censoring is not a no-op (i.e., that the safe and leaky event sets actually differ—catching the failure mode where a broken censor would silently produce a plausible-looking but wrong graph with no error), and one property-style test across multiple cutoff times.

Added `meds_reader` as a new optional dependency extra (`data`) in `pyproject.toml`, kept out of the base install/dependencies list since it's only needed when actually touching real data.

**Code and packages touched:**
- `src/ehrgraph/data/loader.py` (new) — censoring logic and meds_reader wrapper
- `tests/fixtures/make_synthetic.py` (new) — synthetic fixture with no real patient data
- `tests/test_censoring.py` (new) — 3 tests for censoring correctness and boundary conditions
- `tests/__init__.py` (new) — empty, makes tests a package for imports
- `pyproject.toml` — added `data = ["meds_reader"]` optional dependency group
- `notebooks/00_setup_and_sanity_check.ipynb` — added cell opening real meds_reader database and printing live events; updated pip install line to `pip install -e ".[dev,data]"`
- `notebooks/README.md` — same pip install line update
- New dependency: `meds_reader` — optional/extras-only (not in base package dependencies)

**Verification:** `ruff check .` clean, `pytest -q` passing 4/4 (all synthetic-fixture based; includes pre-existing smoke test). Real-data verification (opening the real meds_reader database, printing real events for a real subject) could not be run in this environment—the development machine has only 2.9 GB of free disk space, not enough to extract the 4.3 GB real EHRSHOT_ASSETS.zip. A new cell was added to the Colab sanity-check notebook specifically for this, meant to be run by the human collaborator in Google Colab where there's enough disk space, as the last step before merging the PR.

**Follow-ups / open questions:** PR #21 needs the real-data cell run in Colab and confirmed before merging. This issue also became a head start on issue #9 (synthetic fixture + full leakage test suite), which will build further on the same fixture file rather than starting from scratch.

---

## #3 — Validate locked benchmark tasks' labels and splits, build vocabulary and cohort modules
**Date:** 2026-09-21 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/26 · **Branch:** issue-3-labels-splits-exploration

**What happened:** Validated the two locked benchmark tasks' official labels and splits against PROJECT_GUIDE.md using the full official EHRSHOT_ASSETS dataset (not the professor's trimmed copy). guo_los (Long Length of Stay) confirmed at 3855 patients / 1271 ever-positive, and new_hypertension at 2328 patients / 386 ever-positive—both match PROJECT_GUIDE.md exactly. Discovered and validated a critical graph-building insight: label rows outnumber patient counts for both tasks because a patient can have multiple prediction_time rows with different label values, establishing that (patient_id, prediction_time) is the correct node identity for graph construction, not patient_id alone—this directly informs issue #4's build_patient_vectors node keying strategy.

Built three new tested modules: `ehrgraph.data.vocabulary` (resolves raw "VOCAB/CODE" event codes like "ICD9CM/623.8" to human-readable disease names via the full official OMOP concept.csv, a 5.9M-row table in EHRSHOT_ASSETS/femr/logs/omop_dir/, achieving 100% match rate across all distinct condition codes in both cohorts); `ehrgraph.data.cohort` (code-frequency counting and demographic summaries—sex, race, age-at-prediction-time—with patient-weighted and event-weighted counts kept separate); and `ehrgraph.eval.labels` (one-row-per-task summary across all 15 released EHRSHOT benchmark tasks for broad dataset understanding before zooming into the two locked tasks). Built and fully executed `notebooks/02_official_labels_and_splits.ipynb` end-to-end against real local data: broad pass across all 15 tasks (cohort sizes, class balance), detailed zoom-in per locked task (class balance by split, temporal coverage, demographics, top diagnoses comparison, PCA projection of CLMBR embeddings, lift-weighted diagnosis co-occurrence network preview). Notable finding: ever-positive patients skew older in both tasks, and guo_los-positive patients show cardiometabolic comorbidity patterns (hypertension, hyperlipidemia, abnormal ECG) more often than negative patients. Notebook explicitly flagged its use of full patient history (not prediction_time-censored) as not leakage-safe, deferring that to issues #6/#7/#8.

Also added an optional `table` field (default None, backward compatible) to the `Event` dataclass in `loader.py` so synthetic test fixtures could mimic real meds_reader events.

**Code and packages touched:**
- `src/ehrgraph/data/vocabulary.py` (new) — code-to-concept-name resolution via concept.csv OMOP join
- `src/ehrgraph/data/cohort.py` (new) — cohort code-frequency and demographic summaries
- `src/ehrgraph/eval/labels.py` (new) — cross-task label statistics
- `src/ehrgraph/data/loader.py` — added optional `table` field to `Event` dataclass
- `tests/fixtures/make_synthetic.py` — added `FakeDatabase` / `cohort_with_conditions()` fixture shaped like real meds_reader output
- `tests/test_cohort.py`, `tests/test_labels.py`, `tests/test_vocabulary.py` (new) — 21 new synthetic-fixture-based tests
- `notebooks/02_official_labels_and_splits.ipynb` (new) — end-to-end benchmark task exploration and validation

**Verification:** `ruff check .` clean, `pytest -q` passing 31/31 (all synthetic-fixture based, no real data in CI). Notebook executed end-to-end via `jupyter nbconvert --execute` against the real full official dataset—zero errors, all assertions passed (patient/positive counts matched PROJECT_GUIDE.md exactly), 100% concept-name match rate across all distinct condition codes in both cohorts, 100% CLMBR-embedding join rate. Notebook outputs stripped via nbstripout before commit. PR #26 opened, all 3 CI checks (lint-and-test, lint-notebooks, no-data-committed) passing. No new dependencies—everything used (pandas, networkx, scikit-learn, matplotlib) was already in pyproject.toml.

**Follow-ups / open questions:** None blocking—ready to merge. Issue #4 (build_patient_vectors) can now proceed knowing the correct node identity is (patient_id, prediction_time).

---

## #4 — Implement patient vectorization module
**Date:** 2026-09-21 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/27 · **Branch:** issue-4-patient-vectorization

**What happened:** Implemented `build_patient_vectors(db, subject_ids, table, weighting, dedup, rollup)` in a new module `ehrgraph.graphs.vectorize`—the critical step between raw patient event data and graph construction, turning each patient's medical-code history into a numeric vector. The function deliberately supports two independent configuration axes (weighting: binary/count/count_normalized, and dedup: per_occurrence/per_visit/per_patient) rather than one locked pipeline, enabling systematic robustness checks. This session proved that binary weighting is mathematically dedup-invariant by construction (verified both via unit test on synthetic fixtures and re-check against the real 3,850-patient guo_los cohort). Built and fully executed `notebooks/03_patient_vectorization.ipynb` end-to-end against the real official dataset, running all 9 (weighting, dedup) combinations and per-relation-type vectorization (condition, drug_exposure, procedure separately), catching a genuine meds_reader bug in the process (Event.table uses internal names like "procedure", not "procedure_occurrence").

**Code and packages touched:**
- `src/ehrgraph/graphs/vectorize.py` (new) — build_patient_vectors and WEIGHTINGS/DEDUPS constants
- `src/ehrgraph/data/loader.py` — added optional `visit_id` field to `Event` dataclass (backward compatible)
- `tests/fixtures/make_synthetic.py` — added `cohort_for_vectorization()` fixture for dedup and rollup testing
- `tests/test_vectorize.py` (new) — 12 synthetic-fixture-based tests
- `notebooks/03_patient_vectorization.ipynb` (new) — end-to-end real-data verification of all 9 combinations and per-relation-type check

**Verification:** `ruff check .` clean, `pytest -q` passing 43/43. Notebook executed end-to-end twice via `jupyter nbconvert --execute` against the real full official dataset—zero errors both times. Real matrix sizes confirmed: condition (3850 patients x 16,263 codes, 277,926 nonzero), drug_exposure (3850 x 5,407, 294,749 nonzero), procedure (3850 x 9,087, 189,095 nonzero) after table-name bug fix. Notebook outputs stripped via nbstripout before commit. PR #27 opened, all 3 CI checks passing.

**Follow-ups / open questions:** Building a real concept_ancestor-derived rollup map (ontology-aware similarity robustness check) is separate follow-up work, listed in PROJECT_GUIDE.md's "Can improve" section—not blocking. Issue #5 (bipartite patient-code graph, TF-IDF/cosine projection, mutual k-NN sparsify) can now proceed using this module's output.

---

## #5 — Build patient-similarity graph with TF-IDF and k-NN sparsification
**Date:** 2026-09-21 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/28 · **Branch:** issue-5-similarity-graph

**What happened:** Built the actual patient-similarity graph—the first real graph this project has constructed; everything before this (issues #1–#4) was groundwork. Created a new module `ehrgraph.graphs.similarity` with a clear pipeline: `to_tfidf` (applies IDF weighting on the binary vectors from issue #4), `cosine_similarity_matrix` (computes patient-to-patient similarity, staying sparse with diagonal zeroed), `top_k_sparsify` (keeps each patient's 10 most similar neighbors), `mutual_knn` (primary method: keeps edges only if both patients picked each other), `union_knn` (secondary method: keeps edges if either patient picked the other), `to_networkx` (builds the graph object), and `graph_summary` (reports counts and connectivity). Importantly, IDF weighting is applied on *binary* presence (not raw counts), preserving the issue-#4 decision to avoid conflating rare diagnoses with frequently-recorded ones. Built and executed `notebooks/04_similarity_graph_construction.ipynb` end-to-end against the real 3,850-patient guo_los cohort, finding that mutual k-NN yields 245 isolated patients (6.4%) and 92.6% in the largest component, while union k-NN isolates only 12 patients (0.3%) with 99.7% in one giant component—this gap itself is an informative finding on diagnosis-pattern clustering.

**Code and packages touched:**
- `src/ehrgraph/graphs/similarity.py` (new) — to_tfidf, cosine_similarity_matrix, top_k_sparsify, mutual_knn, union_knn, to_networkx, graph_summary
- `tests/test_similarity.py` (new) — 14 synthetic-fixture-based tests covering IDF behavior, cosine edge cases, sparsification asymmetry, reciprocation logic, and component counting
- `notebooks/04_similarity_graph_construction.ipynb` (new) — end-to-end real-data pipeline execution

**Verification:** `ruff check .` clean, `pytest -q` passing 53/53. Notebook executed end-to-end three times via `jupyter nbconvert --execute` against the real full official dataset (after ruff format and import-order auto-fix)—zero errors every time. Real numbers confirmed: 58.99% density before sparsification, 38,353 directional entries kept by top-10, 9,091 undirected edges after mutual k-NN, 29,262 after union k-NN. Notebook outputs stripped via nbstripout before commit. PR #28 opened, all 3 CI checks passing.

**Follow-ups / open questions:** None blocking. This notebook uses full patient history (no prediction_time censoring)—intentional, deferred to issues #6/#7/#8.
