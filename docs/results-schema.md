# Results schema — `results/aggregate/` (frozen, issue #18)

**Version 1.0, frozen 2026-10-02.** Every file in `results/aggregate/` follows this schema. It is the *only* thing the report figures (`r-viz/`), the eventual site, and the paper read, and the only project output that leaves the machine (see `DATA_ACCESS.md`).

Two things enforce it:
- **Writing:** `ehrgraph.eval.aggregate.write_aggregate` refuses rows that break it.
- **Committed files:** `tests/test_results_schema.py` re-checks every committed CSV in CI, which needs no patient data.

Changing the schema means bumping the version here *and* in `ehrgraph.eval.aggregate.SCHEMA_VERSION`, in the same pull request.

## Columns

One row = one number about one graph (or graph ensemble) for one task. **Never one row per patient.**

| Column | Type | Meaning |
|---|---|---|
| `task` | text | `guo_los` (Task A, Long Length of Stay), `new_hypertension` (Task B), or `population` (whole-population graphs, e.g. the disease-trajectory graph) |
| `condition` | text | Which graph the number describes (see below) |
| `graph_variant` | text | How the graph was built (see below) |
| `metric` | text | What was measured (see below) |
| `value` | number | The measurement. May be empty when undefined (e.g. a null-ensemble metric with no defined seed) |
| `ci_low`, `ci_high` | number, optional | For `null_model` rows: the 2.5th and 97.5th percentiles across the null ensemble. For `homophily_era_only`: the same across label shuffles |
| `seed` | integer, optional | Set only for per-node summaries taken from one specific null graph (seed 0). Empty for real graphs and ensemble summaries |

Patient-identifying columns (`patient_id`, `subject_id`, `person_id`, `omop_person_id`, `prediction_time`, `node`, `node_id`) are refused outright.

## `condition`

| Value | Meaning |
|---|---|
| `safe` | Leakage-safe graph: diagnoses before each prediction time |
| `leaky` | Intentionally leaky graph: each patient's whole record |
| `null_model` | Degree-preserving, weight-shuffled rewiring of `safe`. Ensemble mean, with `ci_low`/`ci_high` (not `null`, which pandas reads as missing). In `spreading.csv`, one rewired graph with `seed` = 0 |
| `reference` | Facts about the cohort, not a graph (e.g. `prevalence`) |
| `leaky_vs_safe` | A comparison of the two graphs (e.g. `edge_jaccard`) |
| `shared_edges`, `safe_only_edges`, `leaky_only_edges` | Links in both graphs, only in safe, only in leaky |
| `raw_codes`, `snomed_harmonized` | Disease-trajectory graph on raw billing codes vs. SNOMED-mapped concepts |

## `graph_variant`

`<codes>_<weighting>_<dedup>_<similarity>_<sparsifier>_k<k>[_<suffix>]` for patient graphs. For example:

| Value | Meaning |
|---|---|
| `omop_concept_binary_perpatient_tfidf_mutual_k10` | **Primary since Oct 2 2026.** Course OMOP tables: standard SNOMED concepts, binary presence per node, TF-IDF + cosine, mutual 10-NN, labels rebuilt from EHRSHOT's definitions |
| `omop_source_binary_perpatient_tfidf_mutual_k10` | Same, on the original ICD-9/ICD-10 billing codes (keeps the 2015 coding-era split) |
| `omop_concept_binary_perpatient_tfidf_mutual_k10_official_labels` | Primary graphs with Stanford's official labels (full release needed; not in the committed Project 1 files) |
| `omop_top200_rr365d_support10_bonferroni_symmax` | Disease-trajectory graph from the course tables: top 200 codes, relative risk within 365 days, ≥ 10 patients, Bonferroni, symmetrized by max |
| `cond_binary_perpatient_tfidf_mutual_k10`, `snomed_…`, `…_clean_cohort`, `top200_…` | Pre-Oct-2 variants built from the full release's meds_reader database (no longer in the committed files) |

The Project 2 graph-variant sweep adds values in the same pattern (e.g. `..._union_k20`).

## `metric` families

Names are `snake_case`, outcome-neutral (`positive`/`negative`, never task-specific words), and identical across tasks.

| Family | Metrics | Range / unit |
|---|---|---|
| Size and connectivity | `n_nodes`, `n_edges`, `n_isolated`, `isolated_fraction`, `n_components`, `largest_cc_size`, `largest_cc_fraction` | counts / share |
| Structure | `density`, `mean_degree`, `median_degree`, `max_degree`, `avg_clustering`, `avg_clustering_weighted`, `transitivity`, `degree_assortativity`, `lcc_avg_shortest_path_hops` (estimated from 500 sampled sources), `lcc_diameter_hops_lower_bound` | |
| Centrality | `{degree,betweenness,closeness,eigenvector}_centrality_{mean,median,p90,max}`, `..._auroc_vs_label`, `spearman_<a>_vs_<b>` | closeness/eigenvector within the largest component only |
| Communities | `n_louvain_communities`, `n_leiden_communities`, `louvain_modularity`, `leiden_modularity`, `adjusted_rand_index`, `leiden_outcome_{eta_squared,nmi,share_in_reported_communities}` | groups < 20 nodes excluded from outcome summaries |
| Outcome alignment | `edge_homophily`, `class_insensitive_homophily` (Lim et al. 2021; 0 = random), `same_outcome_rate_{positive,negative}`, `dirichlet_energy_clmbr` (centered CLMBR, normalized; lower = smoother) | |
| Leakage and same-patient (#15) | `same_patient_edge_fraction`, `same_patient_edge_same_outcome_rate`, `class_insensitive_homophily_excl_same_patient`, `max_single_code_outcome_gap`, `edge_jaccard`; per edge set `n_edges`, `same_outcome_rate`, `both_positive_rate`, `same_patient_fraction` | |
| Coding era (#35) | `icd9_icd10_cross_edges`, `icd9_icd10_cross_edge_fraction`, `era_assortativity`, `homophily_era_only` | |
| Cohort and labels (#35, #37) | `prevalence`, `n_labels`, `clmbr_coverage` (share of nodes with a CLMBR fingerprint; 0 without the full release); rebuilt-vs-official: `rebuilt_labels`, `official_labels`, `shared_prediction_times`, `shared_share_of_official`, `outcome_agreement`, `rebuilt_prevalence`, `official_prevalence`, `official_{positives,negatives}_already_diagnosed`, `official_{positive,negative}_share_already_diagnosed` | |
| Spreading and partitioning (Project 1 notebook 05) | `sir_ever_infected_t{10,30}` (SIR, beta 0.3, gamma 0.1, 10 starting nodes, mean of 20 runs), `cascade_reached`, `cascade_positive_share` (independent cascade, p = 0.1, from 5% of positives, 50 runs), `spectral_k<k>_{modularity,eta_squared,ari_vs_leiden}` (largest component) | shares / counts |
| Trajectory dendrogram (#14) | `cross_vocabulary_edges`, `gn_levels`, `gn_forced_joins`, `gn_best_n_communities`, `gn_best_modularity`, `louvain_n_communities`, `louvain_modularity`, `ari_gn_best_vs_louvain` | |

## Files

| File | Written by | Contents |
|---|---|---|
| `network_stats_task_a.csv` | notebook 02 | Long stay, primary variant, all patient-graph families above |
| `network_stats_task_b.csv` | notebook 03 | New hypertension, same |
| `trajectory_dendrogram.csv` | notebook 04 | Disease-trajectory graph, raw billing codes vs SNOMED concepts |
| `spreading.csv` | notebook 05 | SIR, cascades, spectral vs Leiden |
| `checks.csv` | notebook 06 | Coding era and billing-code graphs for both tasks |
| `figures/NN_*.png` | notebook NN | Figures built only from the above or from anonymous layouts |

## Changes

- **Oct 16 2026, no schema change:** Project 1 notebooks simplified to 01–06 on the trimmed course tables only. `robustness_checks.csv` replaced by `checks.csv` (no official-label rows); new `spreading.csv`.

- **Oct 2 2026 (#37), no schema change:** primary results now come from the course OMOP tables (new `graph_variant` values above). New metrics `n_labels`, `clmbr_coverage` and the label-comparison family. `dirichlet_energy_clmbr` is absent when the full release isn't available.

### Before the freeze

- `same_outcome_rate_long_stay` / `_not_long_stay` → `same_outcome_rate_positive` / `_negative`, and `long_stay_prevalence` → `prevalence` (#33).
- Trajectory `largest_component_size` → `largest_cc_size`, to match the patient graphs (#18).
- The null condition is written as `null_model`, never `null` (#32).
