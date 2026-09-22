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

---

## #6, #7, #8 — Safe/leaky/null graph conditions and null-graph construction
**Date:** 2026-09-21 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/29 · **Branch:** issue-6-7-8-graph-conditions

**What happened:** Built the three graph conditions that form the core experimental design—the same patient-similarity graph pipeline (issues #4/#5), run three different ways to answer the research question: "does a leakage-safe graph genuinely help prediction, or does it only look like it helps because it's allowed to cheat?" Safe (issue #6): each patient's history is cut off strictly before their own prediction_time—the honest, deployable version. Leaky (issue #7): each patient's entire history, including anything after prediction_time—a deliberate negative control. Null (issue #8): the safe graph's structure with the same degree sequence and edge-weight distribution, but randomly scrambled—tests whether graph helps because of real medical similarity or because any graph-shaped structure would help.

A real architectural finding drove the implementation approach: checking the official guo_los data showed 36.5% of patients have more than one prediction_time row (multiple hospital admissions), each needing its own separate view of "history known so far"—not one shared vector per patient. This confirmed issue #3's insight that (patient_id, prediction_time) is the correct node identity, not patient_id alone. Added a new function `build_node_vectors` to the vectorize module, keyed by (patient_id, prediction_time) pairs. A `censor` boolean flag picks the graph condition by calling directly into the already-tested `events_before`/`events_all` functions from issue #1—so safety-critical time-censoring logic was never reimplemented, only reused. Refactored the underlying counting/weighting logic into a shared private helper `_matrix_from_counts` so both the original per-patient `build_patient_vectors` (issue #4) and the new per-node `build_node_vectors` use the identical, already-tested counting code, just keyed differently.

Built an entirely new module `ehrgraph.graphs.null_model` implementing the null-graph construction specified in PROJECT_GUIDE.md: uses networkx's `double_edge_swap` to randomly rewire the graph's edges while keeping every node's degree exactly identical to the original, then separately reassigns the real graph's edge-weight collection to the swapped edges at random. This deliberately avoids networkx's simpler `configuration_model` function, flagged in the project's planning docs as silently discarding all edge weights—using it would have handicapped the null-graph control by comparing a weighted real graph against an unweighted fake one.

While this work was in progress, also closed out unfinished business from issue #9 (back from issue #1): that issue's test suite originally specified 4 tests but only had 3, missing a "determinism" test (same random seed produces an identical graph, checked via graph hash) because real graph-building code didn't exist yet. Now that issues #6–#8 provide real graphs to hash, that fourth test was written and added to `tests/test_censoring.py`, bringing issue #9's original test suite to its full completion.

Built and executed `notebooks/05_leakage_conditions.ipynb` end-to-end against the real official dataset, using all 6,990 (patient, prediction_time) node pairs from guo_los (3,850 distinct patients). Confirmed issue #7's exact done-when requirement directly with an assertion: the safe and leaky graphs' edge sets are provably different. Confirmed issue #8's exact done-when requirements: the null graph's degree sequence and edge-weight distribution match the real safe graph exactly. Found a genuinely interesting, somewhat counter-intuitive real result reported honestly: even though the leaky graph has 2.13x as much raw information per node (764,665 vs 359,819 nonzero matrix entries), it actually produces a MORE fragmented graph than the safe graph—7.3% of nodes end up isolated vs. 4.4% for safe, and only 61.9% of nodes land in the largest connected cluster vs. 87.4% for safe. The likely explanation documented in the notebook: including a patient's entire lifetime of history accumulates a much longer, more idiosyncratic set of diagnoses, which can make patients look less alike on a simple overlap basis, not more.

**Code and packages touched:**
- `src/ehrgraph/graphs/vectorize.py` — added `build_node_vectors` and shared `_matrix_from_counts` helper (refactored, not duplicated, from existing `build_patient_vectors` logic)
- `src/ehrgraph/graphs/null_model.py` (new) — `build_null_graph` using double_edge_swap + random weight reassignment
- `tests/test_vectorize.py` — 3 new tests for `build_node_vectors` (per-node censoring, leaky ignoring prediction_time, same patient appearing as multiple distinct nodes)
- `tests/test_null_model.py` (new) — 6 new tests (degree sequence preserved, isolated nodes preserved, edge count preserved, weight multiset preserved, edge set actually changes, determinism given seed, handles graphs with too few edges to swap)
- `tests/test_censoring.py` — added the closing determinism test for issue #9 (builds real graph twice with same seed via full pipeline, confirms identical Weisfeiler-Lehman graph hash)
- `notebooks/05_leakage_conditions.ipynb` (new) — end-to-end real-data run of all three graph conditions
- No new dependencies—networkx's `double_edge_swap` and `weisfeiler_lehman_graph_hash` were already available

**Verification:** `ruff check .` clean, `pytest -q` passing 64/64. Notebook executed end-to-end three times via `jupyter nbconvert --execute` against the real full official dataset (once initially, twice more after fixing line-length lint issues) to confirm cleanup passes never silently broke anything—zero errors every time, and identical real numbers reproduced each run. Notebook outputs stripped via nbstripout before commit. PR #29 opened, all 3 CI checks (lint-and-test, lint-notebooks, no-data-committed) passing.

**Follow-ups / open questions:** None blocking. This same `build_node_vectors`/`build_null_graph` machinery is ready to reuse directly for Task B (new_hypertension) in issue #16, and for the network-analysis issues (#11–#15) that compare these three conditions' structural properties in more depth.

---

## Follow-up on #6, #7, #8 — Investigating the leaky-graph fragmentation finding
**Date:** 2026-09-22 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/29 · **Branch:** issue-6-7-8-graph-conditions

**What happened:** After the previous entry's work, the user posed a methodologically skeptical question: if the leaky graph came out MORE fragmented than the safe graph (7.3% isolated nodes vs. 4.4%, 61.9% in largest component vs. 87.4%), does this contradict the intuition that leakage should look "better" and thereby call the whole experiment into question? Rather than reason abstractly, ran diagnostic checks directly against the real data. The initial hypothesis—that leaky's much longer vectors (109 codes/node on average vs. 51 for safe) were "diluting" cosine similarity scores—was tested and found to be wrong: leaky's mean and median nonzero cosine similarities were actually marginally *higher* than safe's (mean 0.0389 vs 0.0385, median 0.0286 vs 0.0270), and 71.3% of all patient pairs had nonzero similarity in leaky vs. only 44.8% in safe. The mechanism was identified as specific to the mutual k-NN sparsification step: leaky's much larger, noisier vectors spread similarity scores more evenly across many candidates (more near-ties, less separation between 10th and 50th-best match), making it statistically less likely that any two patients will land in each other's exact top-10 list, even though raw similarity is more abundant overall. This was tested by building both graphs under union k-NN (keeps edges if *either* patient picks the other, no reciprocity required): the fragmentation gap reversed entirely (99.1% largest-component fraction for leaky vs. 98.8% for safe, only 0.17% isolated nodes for leaky vs. 1.19% for safe), exactly confirming the mechanism. Conclusion: the fragmentation is real and mechanistically understood, specific to the mutual-k-NN choice, and NOT a sign of broken leakage. Critically, it says nothing yet about predictive performance (AUROC, etc.—a separate comparison planned for issues #15/#17). The experiment was judged sound to proceed on this basis.

**Code and packages touched:**
- `notebooks/05_leakage_conditions.ipynb` — added real-data diagnostic section (mean/median nonzero cosine similarity, similarity-matrix density, safe vs. leaky comparison) and union-k-NN robustness-check section; replaced incorrect "dilution" explanation with verified mutual-k-NN-reciprocity mechanism in summary; updated done-when checklist
- No source code changes — investigation and documentation only; `union_knn` already existed from issue #5

**Verification:** `ruff check .` clean, `pytest -q` passing 64/64 (unchanged). Notebook executed end-to-end against the real full official dataset—zero errors, all real diagnostic numbers reproduced exactly (mean codes/node: 51.5 safe vs 109.4 leaky; mean nonzero cosine similarity: 0.0385 safe vs 0.0389 leaky; similarity density: 44.8% safe vs 71.3% leaky; union k-NN largest-CC fraction: 98.8% safe vs 99.1% leaky). Notebook outputs stripped via nbstripout before commit. Pushed as additional commit to still-open PR #29, all 3 CI checks (lint-and-test, lint-notebooks, no-data-committed) still passing.

**Follow-ups / open questions:** None blocking. This investigation strengthens confidence in proceeding as designed. PR #29 merge and issue closure (#6, #7, #8) pending user review and approval before moving to issue #10 (disease-trajectory graph).

---

## #10 — Disease-trajectory graph: connect diagnoses by disease progression patterns
**Date:** 2026-09-22 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/30 · **Branch:** issue-10-disease-trajectory-graph

**What happened:** Built an entirely different kind of graph from everything earlier in this project. Every previous graph connected patients to other patients (based on shared diagnoses); this one connects diagnoses to other diagnoses, based on how often one condition leads to another over time—called a disease-trajectory graph. It's not used for prediction; it's explanatory, meant to visually show disease progression patterns, and it feeds into a future clustering step (issue #14, Girvan-Newman community detection) that needs this kind of graph as input. Built a new module, `ehrgraph.graphs.trajectory`, that implements a graph-construction rule which had never been fully specified before in the project's planning documents. The approach: take the 200 most common diagnoses across the entire patient population (not just one prediction task's cohort, since disease progression is a population-wide pattern, not specific to any one prediction question). For every ordered pair of diagnoses A and B, compute a relative risk: among patients who have A, what fraction go on to develop B within a 1-year window afterward, compared to how common B is in the general population? A relative risk of 3 means "having A make you 3x more likely to get B than a random person." Only keep a connection in the final graph if two conditions are met: enough patients actually showed the pattern (at least 10), and the pattern is statistically real under a formal significance test, made much stricter to account for testing thousands of diagnosis pairs simultaneously (a standard statistical safeguard called Bonferroni correction, which divides the significance bar by the number of comparisons being made—with 200 diagnoses there are 39,800 possible ordered pairs, so the bar gets very strict, roughly 1 in 800,000 instead of the usual 1 in 20). The design choices here (specifically the 1-year time window and the minimum-10-patients threshold) were chosen by first checking what the real data actually looked like under different settings, not picked arbitrarily or copied from a default. The final graph also gets converted into an undirected version (each pair's edge weight becomes the larger of the two possible directions) since the next issue's clustering algorithm requires an undirected graph—this was implemented as its own function with an explicit note that converting to undirected on purpose discards the "which one causes which" information, a real tradeoff worth stating plainly in any eventual report rather than hiding.

Built and executed notebook `notebooks/06_disease_trajectory_graph.ipynb` end-to-end against the real official dataset covering all 6,731 patients in the database. This notebook was written to a different, higher bar than earlier ones specifically per direction from the user: heavily narrated so a reader with no prior background on the project could follow along using only the notebook's own explanations, and with a strong focus on producing polished, genuinely readable visualizations rather than default/quick charts. Real findings surfaced: of the 39,800 possible diagnosis-pair relationships, only 704 survived the full statistical filtering process, and the ones that did survive make real clinical sense (for example, "acute respiratory failure" significantly predicting "pulmonary congestion" afterward, with about a 6x elevated risk). The graph naturally splits into separate disconnected clusters—two large ones (70 and 50 diagnoses each) plus several small isolated pairs that turned out to be essentially duplicate diagnosis codes for the same underlying condition recorded close together in time (for example "Obesity, unspecified" paired with "Morbid obesity", or "Low back pain" paired with "Dorsalgia")—a genuine, interesting artifact of how the same clinical concept sometimes gets recorded under two different billing codes. The diagnoses with the most incoming connections (meaning many other conditions tend to lead to them) were plausible late-stage complications like pulmonary congestion and chronic pulmonary edema, which is a real sanity check that the graph is capturing genuine clinical relationships rather than statistical noise. A significant amount of iteration went into the visualizations specifically, catching real problems by actually rendering and looking at the images rather than assuming the code was correct: the first version used a standard default network layout algorithm on a graph that wasn't fully connected, which wasted most of the picture on empty space and crammed the actual clusters into dense, unreadable blobs with overlapping text labels. This was fixed by detecting the graph's separate connected pieces first and giving each one its own properly-sized panel with its own layout, plus by trimming an overly dense 309-edge directional view down to just the 40 strongest relationships so individual arrows could actually be followed by eye.

**Code and packages touched:**
- `src/ehrgraph/graphs/trajectory.py` (new) — first_occurrence_times, compute_relative_risks, bonferroni_significant, build_trajectory_graph, symmetrize_max
- `tests/test_trajectory.py` (new) — 11 new tests, all synthetic-fixture based, no real data needed in CI
- `notebooks/06_disease_trajectory_graph.ipynb` (new) — end-to-end real-data run with heavy narrative documentation and polished visualizations

**Verification:** `ruff check .` clean, `pytest -q` passing 74/74. Notebook executed end-to-end multiple times via `jupyter nbconvert --execute` against the real full official dataset (initial build, two visualization redesign passes, one after lint-driven formatting changes) to confirm nothing was silently broken along the way—zero errors every time, with identical real numbers reproduced on every run. Notebook outputs stripped via nbstripout before commit. PR #30 opened, all 3 CI checks (lint-and-test, lint-notebooks, no-data-committed) passing.

**Follow-ups / open questions:** None blocking. This graph and module are ready to feed directly into issue #14 (Girvan-Newman dendrogram), which needs exactly this undirected, weighted trajectory graph as its input.
