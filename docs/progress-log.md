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

---

## Sep 25 audit — device-only data rule, Task B baseline verified, aggregate writer
**Date:** 2026-09-25 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/31 · **Branch:** chore/sep25-audit

**What happened:** Team decision established that the EHRSHOT dataset stays strictly on the local device (no Colab, Google Drive, or Redivis export); GitHub carries only code, notebooks with outputs stripped, and results/aggregate/ (summary CSVs plus figures). Updated DATA_ACCESS.md, README.md, and notebooks/README.md with this policy, removing stale "Colab workflow" sections. Hypertension (Task B) baseline now verified: notebook 01 previously printed it without asserting, now asserts |diff| < 0.02, confirming AUROC 0.7125 vs Stanford's published 0.71758 (diff 0.005, C=0.0003). New module src/ehrgraph/eval/aggregate.py with write_aggregate() enforces the long-format schema, refuses patient-identifying columns and paths outside results/aggregate/, and caught a real bug where the condition name "null" was silently turned into NaN by pd.read_csv. Repo health checks found pre-commit hooks not installed (now installed) and working-copy edits to notebooks that were older versions (stashed, not committed).

**Code and packages touched:**
- `src/ehrgraph/eval/aggregate.py` (new) — write_aggregate with schema enforcement and validation
- `DATA_ACCESS.md`, `README.md`, `notebooks/README.md` — updated with device-only policy, removed stale Colab sections
- `tests/test_aggregate.py` (new) — 6 tests
- No change to `.pre-commit-config.yaml` itself — its hooks were simply never installed in the main local clone; now they are (`pre-commit install`)

**Verification:** pytest passing, ruff clean, notebook 01 executed end to end with both assertions passing; CI checks on PR #31 passing.

**Follow-ups / open questions:** User should confirm the old Google Drive copy (ehrshot_transfer folder) was deleted and the Drive trash emptied, per Stanford's Sep 20 reply. GitHub Student Developer Pack still worth confirming before the Week 4 placeholder site.

---

## #11, #12, #13 — Network statistics for Task A: structure, centrality, communities, homophily
**Date:** 2026-09-25 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/32 · **Branch:** issue-11-12-13-network-stats-task-a

**What happened:** Built notebook 07_network_stats_task_a.ipynb comparing safe, leaky, and null graphs for Long Length of Stay (6,990 admissions, prevalence 0.253) against a 10-seed null ensemble. Key findings: transitivity 0.498 (safe) vs 0.660 (leaky) vs 0.001 (null); average path length 9.6 vs 11.9 vs 5.3 hops; Leiden modularity 0.942 vs 0.980 vs 0.558; share of outcome variance explained by community 0.121 vs 0.147 vs 0.007. Real graphs are strongly homophilous and aligned with CLMBR far beyond chance; leakage changed graph shape but not outcome alignment. Centrality does not predict outcome (AUROC 0.47–0.55). Fixed four bugs: (1) class-insensitive homophily was missing base-rate term, replaced with Lim et al. formula; (2) switched networkx betweenness/closeness to igraph's C implementation (~13 s per graph vs 14–22 min); (3) eigenvector centrality now computed in largest connected component only; (4) average path length now estimated by BFS from 500 samples (9.559 vs exact 9.558, 16 s vs 189 s).

**Code and packages touched:**
- `src/ehrgraph/netstats/centrality.py`, `communities.py` (adds modularity), `homophily.py` (corrected class_insensitive_homophily), `structure.py` (new: structural_stats, average_path_length_sampled) — network statistics implementations
- `src/ehrgraph/graphs/similarity.py` (new build_similarity_graph) — vectors-to-graph pipeline previously inlined in notebook 05
- `tests/test_centrality.py`, `test_communities.py`, `test_homophily.py`, `test_structure.py`, `test_similarity.py` (new/updated) — 105 total tests
- `notebooks/07_network_stats_task_a.ipynb` (new) — results/aggregate/network_stats_task_a.csv and 5 figures

**Verification:** pytest 105 passed, ruff clean, notebook executed end to end three times against real on-device data; final run produced committed CSV and figures; numbers cross-checked against CSV.

**Follow-ups / open questions:** #14 (Girvan-Newman dendrogram), #15 (safe-vs-leaky-vs-null structural comparison), #16 (repeat for Hypertension), #18 (freeze aggregate CSV schema). Notebook 05 could later be switched to build_similarity_graph instead of inlining.

---

## #15, #16, #17 — Leakage deep dive, Hypertension network analysis, Task A vs Task B
**Date:** 2026-10-02 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/33 · **Branch:** issue-15-16-17-leakage-deep-dive-and-task-b

**What happened:** Refactored notebook 07's analysis code into reusable library modules—`src/ehrgraph/netstats/task_analysis.py` (build_task_graphs, load_task_graphs, analyze_task, summary_table, plus helpers same_patient_stats, edge_set_stats, code_outcome_gaps) and `src/ehrgraph/viz/network_figures.py`—so both tasks now run identical code from a shared library, enabling Project 2's graph-variant sweep to reuse it without duplication. Regression check: re-running Task A through the library reproduced all 169 previously committed values exactly, confirming refactoring correctness. Built comprehensive new issue #15 checks that apply to both tasks: share of links joining two admissions of the same patient, homophily with and without same-patient links, which links leakage adds or removes (shared / safe-only / leaky-only comparison), and strongest single diagnosis code's positive-vs-negative prevalence gap.

For Task A (Long Length of Stay): same-patient links account for 37.5% of safe-graph links, agree on outcome 68% of the time (chance baseline 62%), and removing them actually *raises* homophily from 0.209 to 0.237 in the safe graph (from 0.201 to 0.242 in leaky). The leaky-only links join same-outcome patients 68.5% of the time versus 71.7% for the safe-only links they replace—leakage changes graph shape but adds no outcome signal for long stay prediction.

Built entirely new notebook 10 (Task B, Hypertension): 3,760 discharges from 2,324 patients, prevalence 0.137. Homophily safe 0.282 / leaky 0.332 / null 0.004; without same-patient links this drops to 0.136 / 0.193 (leaky 42% above safe once same-patient links are removed). Community structure (eta-squared) 0.076 safe / 0.123 leaky. The leaky vectors contain ICD-9 401.9 (essential hypertension) in 67% of positives versus 5% of negatives (prevalence gap 0.62), compared to just 0.12 for safe. Same-patient links comprise 31% of all links and agree on outcome 96% of the time—they carry roughly half of the safe graph's homophily. However, the graph dilutes even this near-answer-key outcome information (+0.05 homophily signal) because TF-IDF down-weights common codes like hypertension that appear in most patients.

Built notebook 09 (new): direct Task A vs Task B comparison, reading only aggregate CSVs (no patient data access needed). Conclusion: leakage is mechanistically inert for long stay (no outcome signal added) and visibly present for hypertension (same-patient links carry outcome information) *because hypertension itself is a diagnosis code*, making same-patient links inherently informative for this specific prediction task.

Performance optimization: refactored `structure.py` to run BFS on a copied largest connected component instead of a networkx subgraph view object, achieving 7x speedup (27 s down to 4 s on a 7,000-node graph) with identical results. Found during live profiling with py-spy when the first Task A run exceeded 21 minutes; final timings: Task A 11 min, Task B 8 min.

Metric renames for cross-task generality: `same_outcome_rate_long_stay` → `same_outcome_rate_positive` (and `_negative`), `long_stay_prevalence` → `prevalence`.

**Code and packages touched:**
- `src/ehrgraph/netstats/task_analysis.py` (new) — build_task_graphs, load_task_graphs, analyze_task, summary_table, same_patient_stats, edge_set_stats, code_outcome_gaps
- `src/ehrgraph/viz/network_figures.py` (new) — shared network visualization pipeline for both tasks
- `src/ehrgraph/netstats/structure.py` — refactored BFS on copied component (7x faster)
- `tests/test_task_analysis.py` (new) — 5 tests for cross-task analysis functions
- `notebooks/07_network_stats_task_a.ipynb` — rebuilt using new task_analysis library
- `notebooks/09_task_comparison.ipynb` (new) — Task A vs Task B aggregate CSV comparison
- `notebooks/10_network_stats_task_b.ipynb` (new) — complete Hypertension network analysis
- `notebooks/README.md` — updated with Task B instructions
- `results/aggregate/network_stats_task_a.csv` — 193 rows
- `results/aggregate/network_stats_task_b.csv` (new) — Hypertension results
- 14 figures across notebooks 09 and 10
- No new dependencies

**Verification:** `pytest` 110 passed, `ruff` clean, notebooks 07 and 10 executed end-to-end against on-device data with all findings numbers cross-checked against the aggregate CSVs, notebook 09 executed against CSVs only (no patient data needed), CI passing on PR #33.

**Follow-ups / open questions:** Decide how to treat same-patient links before Project 2 begins—report every homophily number both with and without same-patient links; consider a no-same-patient graph variant in the sweep. 9% of Hypertension positives already have an I10 code before prediction_time in safe vectors—may reflect how EHRSHOT defines "new" diagnoses but not yet verified. Remaining Project 1 issues: #14 (Girvan-Newman dendrogram), #18 (freeze results schema), #19 (report), #20 (submit).

---

## #14 — Girvan-Newman dendrogram on the disease-trajectory graph
**Date:** 2026-10-02 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/34 · **Branch:** issue-14-girvan-newman-dendrogram

**What happened:** Built the final network-analysis stage for the trajectory graph from issue #10, running Girvan-Newman edge-betweenness clustering all the way to complete fragmentation, then converting the full split sequence into a scipy linkage matrix and picking the most modular level. The linkage-matrix conversion was the "fiddly" part issue #10's planning had warned about—it required reconstructing parent node identities across merges and handling disconnected pieces by joining them synthetically above every real merge. Added vocabulary.py's standard_concept_map to unify raw billing codes into SNOMED concepts: maps every source code to its OMOP standard concept via the shipped concept_relationship "Maps to" (ICD-9 401.9 and ICD-10 I10 both → SNOMED 59621000). This mapping revealed a critical artifact: on raw billing codes, the trajectory graph's two large pieces were the two coding systems themselves—70 codes all ICD-10 and 50 codes all ICD-9, with zero of 588 links between them, because the US switched from ICD-9 to ICD-10 in October 2015, and a one-year window rarely spans the switch.

After mapping to SNOMED (17,828 codes → 8,429 concepts), the graph unified into one main piece of 59 concepts (75 concepts and 135 links in total). Girvan-Newman produced 15 communities at modularity 0.536; Louvain achieved 12 communities at modularity 0.562 with ARI 0.57 agreement. The communities made real clinical sense: sepsis/acute renal failure, heart failure/cardiomyopathy, respiratory failure/pneumonia, coronary disease, atrial arrhythmias, pleural disease. Built notebook 08 with results/aggregate/trajectory_dendrogram.csv and 3 figures.

**Code and packages touched:**
- `src/ehrgraph/netstats/dendrogram.py` (new) — runs Girvan-Newman to full fragmentation, converts split sequence to scipy linkage matrix, finds most modular level, handles disconnected pieces
- `src/ehrgraph/viz/dendrogram_figures.py` (new) — visualization for dendrogram results
- `src/ehrgraph/graphs/trajectory.py` — build_population_trajectory_graph with optional code_map
- `src/ehrgraph/data/vocabulary.py` — standard_concept_map for OMOP mapping
- `tests/test_dendrogram.py` (new) — 4 tests
- `notebooks/08_trajectory_dendrogram.ipynb` (new) — end-to-end analysis with dendrogram visualization

**Verification:** `pytest` 116 passed, `ruff` clean, notebook 08 executed end-to-end (about 1 minute) against on-device data. FakeDatabase in tests now iterable like meds_reader.

**Follow-ups / open questions:** SNOMED harmonization should become a core arm of the Project 2 graph-variant sweep—the patient graphs show the same era split, quantified in #35.

---

## #18, #35 — Results schema frozen; robustness checks (coding era, SNOMED graphs, clean cohort)
**Date:** 2026-10-02 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/36 · **Branch:** robustness-checks-and-schema

**What happened:** Issue #18 froze the results/aggregate format at version 1.0 with docs/results-schema.md documenting columns, allowed condition and graph_variant values, metric families with meaning and range, and complete file inventory. tests/test_results_schema.py re-checks every committed results CSV in CI against the schema with no patient data access. Set aggregate.SCHEMA_VERSION = "1.0". Renamed trajectory metric largest_component_size to largest_cc_size for consistency with patient graphs. Issue #35 added notebook 11_robustness_checks.ipynb with three critical checks showing every Project 1 conclusion survives.

Coding era check: only 3 of 18,698 long-stay safe links and 6 of 10,535 hypertension safe links join an ICD-9-only patient to an ICD-10-only patient (era assortativity 0.74 and 0.77). Shuffling outcomes within each era left homophily of only 0.004 (long stay) and 0.019 (hypertension) against actual 0.209 and 0.282, so era explains at most 2% and 7% of the observed homophily. SNOMED-harmonized patient graphs: cross-era links rose to 3-4%. Long stay remained unchanged (safe/leaky homophily 0.212/0.205 vs raw 0.209/0.201). Hypertension safe/leaky 0.292/0.324 vs raw 0.282/0.332, shrinking the leakage gap from +0.050 to +0.032 despite the leaky answer-key single-code gap rising from 0.62 to 0.84—TF-IDF down-weights the merged, very common concept. Clean hypertension cohort using EHRSHOT's own definition (SNOMED 59621000 plus its 16-concept subtree via new descendant_concepts function) found 55 of 515 positives (10.7%) and 84 of 3,245 negatives (2.6%) already had an essential-hypertension code before prediction_time, median 14 days earlier. Without those 139 discharges (3,621 left, prevalence 0.127), homophily safe/leaky is 0.258/0.332 (gap +0.074), and without same-patient links 0.108/0.188 (+74%).

**Code and packages touched:**
- `docs/results-schema.md` (new) — version 1.0 specification
- `src/ehrgraph/eval/aggregate.py` — SCHEMA_VERSION = "1.0"
- `tests/test_results_schema.py` (new) — schema validation in CI
- `src/ehrgraph/data/vocabulary.py` — MappedConditions made public, descendant_concepts function
- `src/ehrgraph/netstats/task_analysis.py` — load_task_graphs exclude option, coding_era, era_stats
- `notebooks/11_robustness_checks.ipynb` (new) — three robustness checks end-to-end
- `results/aggregate/robustness_checks.csv` (new) — 615 rows of check results
- `results/aggregate/11_robustness_homophily.png` (new) — figure

**Verification:** `pytest` 129 passed, `ruff` clean, notebook 11 executed end-to-end against on-device data (about 11 minutes). All finding numbers cross-checked against the CSVs.

**Follow-ups / open questions:** Why EHRSHOT's exclusion missed the 139 prior-coded discharges is not established (the shipped vocabulary does map I10 and 401.9 into the subtree)—a careful question to Stanford's team, not a claim. Remaining Project 1 issues: #19 (report) and #20 (submit by Oct 23).

---

## #37 — Run Project 1 on the course OMOP tables; CLMBR optional; publish executed notebooks
**Date:** 2026-10-02 · **PR:** https://github.com/patient-graph-lab/ehrshot-patient-graphs/pull/38 · **Branch:** trimmed-omop-data-source

**What happened:** Team members have the course's trimmed OMOP tables but not Stanford's full EHRSHOT release, preventing them from reproducing Project 1 results. On Oct 2, team decided to make the course tables the primary data source, with CLMBR-dependent steps skipped cleanly for teammates without the full release. Built `src/ehrgraph/data/omop.py` containing OmopTables and OmopDatabase, which present the course CSVs in the same shape as the meds_reader database, so the entire existing pipeline runs unchanged. Two code modes: standard SNOMED concepts (primary, "OMOP/<concept_id>") or original billing codes ("ICD9CM/...", "ICD10CM/..."). Built `src/ehrgraph/data/omop_labels.py` with outcome labels rebuilt from EHRSHOT's published definitions. Long stay: inpatient visits not same-day, prediction at 23:59 on admission day, positive if >= 7 days. New hypertension: inpatient visits not same-day, prediction at 23:59 on discharge day, skipped if already diagnosed (SNOMED 59621000 or descendants), positive if first diagnosed within 365 days. Against Stanford's official labels, long stay shares 6,930 of 6,995 prediction times (99.1%) with 99.84% outcome agreement; hypertension shares 3,499 of 3,764 (93.0%) with 99.40% agreement—the hypertension gap is 139 already-diagnosed discharges Stanford keeps (55 of 516 positives, 84 of 3,248 negatives) plus about 115 same-day visits. Built `src/ehrgraph/data/source.py` and `scripts/local_setup.ensure_data_sources()` making course tables required (EHRSHOT_TRIMMED_ROOT) and full release optional (EHRSHOT_DATA_ROOT). Refactored `src/ehrgraph/netstats/task_analysis.py` with new `task_graphs_from_labels` working on any label table and any database; CLMBR is optional, Dirichlet energy computed only on fingerprinted nodes and skipped entirely without the full release. Executed notebooks 07-11 with outputs committed to `notebooks/executed/`. Built `src/ehrgraph/eval/publish.py` and `scripts/publish_executed.py` that check every output against real patient ids and prediction times on the publishing machine before writing. Built `scripts/check_executed_notebooks.py` running in pre-commit hook and data-guard CI job—no patient-id columns, no tables over 100 rows, no minute-resolution timestamps, no id-shaped 9-10 digit numbers, max 5 MB. Tested: planted real patient ids caught by both checks. Re-ran results on course tables showing same conclusions as before: long stay homophily safe/leaky/null 0.211/0.201/0.010; hypertension 0.226/0.263/0.002, and without same-patient links 0.115/0.154. Notebook 08 billing codes split into 69 ICD-10 and 47 ICD-9 codes with 0 of 584 links between them; SNOMED concepts give one piece of 60 concepts, best cut 13 communities with modularity 0.566. Notebook 11 shows era alone explains at most 2% (long stay) and 6% (hypertension) of homophily; conclusions hold with billing codes and with official labels; hypertension leakage gap ranges from +0.038 to +0.056 across versions. Teammate simulation: notebook 10 run in separate repo copy with no full release ran cleanly, skipped CLMBR (0 of 3,511 fingerprints), gave identical homophily 0.2255. Updated docs: DATA_ACCESS.md (who needs which data), README.md (setup for teammates), notebooks/README.md (which notebook needs which data; notebooks 00-06 remain full-release development history), docs/results-schema.md (new graph_variant values and metrics).

**Code and packages touched:**
- `src/ehrgraph/data/omop.py` (new) — OmopTables and OmopDatabase presenting course CSVs in meds_reader shape
- `src/ehrgraph/data/omop_labels.py` (new) — outcome labels rebuilt from EHRSHOT definitions
- `src/ehrgraph/data/source.py` (new) — course/full release data source handling
- `src/ehrgraph/eval/publish.py` (new) — real patient id/prediction time validation before publishing
- `src/ehrgraph/netstats/task_analysis.py` — task_graphs_from_labels with optional CLMBR, new n_labels and clmbr_coverage rows
- `src/ehrgraph/viz/network_figures.py` — shared visualization pipeline
- `scripts/local_setup.py` — ensure_data_sources with optional/required flagging
- `scripts/publish_executed.py`, `scripts/check_executed_notebooks.py` (new) — publishing and privacy scanning
- `.pre-commit-config.yaml`, `.github/workflows/data-guard.yml` — privacy checks in pre-commit and CI
- `tests/test_omop.py` (7 tests), `tests/test_publish.py` (6 tests) — new tests
- `notebooks/07-11` — executed notebooks with outputs
- `results/aggregate/` — CSVs and figures
- `docs/DATA_ACCESS.md`, `README.md`, `notebooks/README.md`, `docs/results-schema.md` — documentation updates
- No new dependencies

**Verification:** `pytest` 142 passed, `ruff` clean, notebooks 07, 08, 10, 11 executed end-to-end on real data and 09 on CSVs, finding numbers cross-checked against CSVs, all executed notebooks passed exact privacy scan.

**Follow-ups / open questions:** Teammates should check whether their course download includes concept.csv (without it, concepts show as IDs rather than names). Teammates request their own full-release access before Project 2. Project 1 report (#19) and submission (#20) remain.
