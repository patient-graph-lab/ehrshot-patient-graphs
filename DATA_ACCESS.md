# Data Access

This repository contains **no patient data of any kind**, at any level of aggregation finer than the summary statistics in `results/aggregate/`. This is a hard rule, enforced by `.gitignore`, pre-commit hooks, and a CI gate (`data-guard`) on every pull request — see `.github/workflows/data-guard.yml`.

## What this project uses

- **EHRSHOT** (Wornow et al., NeurIPS 2023 Datasets & Benchmarks, [arXiv:2307.02028](https://arxiv.org/abs/2307.02028)) — de-identified EHR data for 6,739 patients, distributed by Stanford via Redivis under a Data Use Agreement (DUA).
- This project uses the **MEDS** format (via `meds_reader`) and the **CLMBR-T-base** frozen embeddings shipped in `EHRSHOT_ASSETS/`.

## Getting your own access

1. Follow Stanford's EHRSHOT access process — see the [`ehrshot-benchmark`](https://github.com/som-shahlab/ehrshot-benchmark) repository for current instructions.
2. **Each collaborator must obtain their own DUA-approved access.** Repository access conveys no rights to the data. Do not share your copy of the data with another team member, including other members of this project — each person must go through the access process independently.
3. Once you have access, point the codebase at your own local copy — never a relative path from the repo root:
   ```bash
   export EHRSHOT_DATA_ROOT=/path/to/your/local/ehrshot/copy
   ```
4. Verify your copy matches what the team is using by comparing against `data/checksums.txt` (`sha256sum` each file) — this lets two collaborators confirm they have the same dataset version without either of them transmitting the data itself.

## Where the data lives: this device only (team decision, Sep 25, 2026)

**The dataset stays on the collaborator's own machine, and nowhere else, until the team explicitly decides otherwise.** That means `~/ehrshot-data/` (or wherever your own `EHRSHOT_DATA_ROOT` points) on a machine you control — no Google Drive, no Colab upload (chunked or otherwise), no Redivis export, no S3, no shared folder. Every notebook runs locally via `scripts/local_setup.py`; Project 1 needs nothing more (the largest graphs have ~7,000 nodes -- one per admission, not per patient).

**GitHub holds code and aggregate results only.** Concretely, the only things that leave the machine are:
- source code, tests, and notebooks **with outputs stripped** (`nbstripout` runs in pre-commit and CI's `notebook-lint` re-checks it — printed outputs can quietly contain patient IDs, so they never get committed);
- summary tables in `results/aggregate/*.csv` (one row per metric, never one row per patient);
- figures in `results/aggregate/figures/` built only from those summaries (distributions, bar charts, anonymous network layouts — no patient IDs, no labels on individual nodes).
- **executed notebooks** in `notebooks/executed/` (outputs kept, team decision Oct 2, 2026), so team members without CLMBR access or the compute can read every result without running anything. They are published **only** through `scripts/publish_executed.py`, which checks every output against the real patient IDs and prediction times on the publishing machine and refuses anything that matches. CI re-scans them on every pull request without data: no ID-shaped numbers, no patient-level columns, no tables over 100 rows, no prediction-time timestamps, at most 5 MB each.

`scripts/colab_setup.py` and `scripts/dismantle_data.py` are still in the repo and still tested, but they are **not to be used** under this policy — they exist only in case the team later reverses the decision (issue #23 tracks Redivis as the only option Stanford actually recommended for cloud compute).

**RESOLVED, Sep 20, 2026 — Google Drive is explicitly disallowed. This is confirmed, not a judgment call.** Stanford's data stewards (shahlab-contact@stanford.edu) replied to the question sent earlier the same day: *"Generally only institution-level Google Drive setups (e.g., Stanford's) which explicitly include citable, documented guidance on protocols for storing high-risk data are permitted. The general rule is to not store the data in Google Drive, since it doesn't meet our explicit requirements around security."* This supersedes every earlier Drive-related risk analysis in this file's history (all now void) — a personal or team Google Drive, brief pass-through or persistent, does not meet Stanford's bar. **Any standing copy in Drive must be deleted immediately, Trash emptied too** (Drive's delete only moves to Trash by default).

## Two data sources, and who needs which (team decision, Oct 2, 2026)

| Source | What it holds | Who needs it |
|---|---|---|
| **Course OMOP tables** (the class's Redivis release: `person`, `visit_occurrence`, `condition_occurrence`, ..., `concept_ancestor`) | Patients, visits, diagnoses (with standard SNOMED concepts and original billing codes), vocabulary | **Everyone.** All Project 1 results are built from these, with outcome labels rebuilt from EHRSHOT's published definitions (`ehrgraph.data.omop_labels`). |
| **Stanford's full EHRSHOT release** (`EHRSHOT_ASSETS`) | CLMBR fingerprints, official labels and splits, published benchmark scores | **Optional for Project 1:** without it only the baseline check (notebook 01) and Dirichlet energy are skipped. **Required for Project 2**, whose models take CLMBR fingerprints as input. |

Point the code at your copies with `EHRSHOT_TRIMMED_ROOT` (course tables; default `~/ehrshot-data/trimmed`) and `EHRSHOT_DATA_ROOT` (full release; default `~/ehrshot-data`). Each person needs **their own** approved access to each source they use. **CLMBR fingerprints and label files are patient-level: never send them to a teammate**, even one who is on the project. A teammate without the full release should request it through their own Redivis account.

## What must never happen

- No raw data file, and nothing derived from it that could re-identify or expose patient-level records, may ever be committed to this repository or published on the project's static site — **regardless of repository visibility**. GitHub Pages sites are publicly reachable even when the source repository is private; repo privacy is not a safeguard for anything published to Pages.
- **No collaborator may upload the data to Google Drive, under any circumstances — confirmed disallowed by Stanford, Sep 20, 2026.** This is no longer a judgment call to make independently.
- No collaborator may upload the data to any other third-party cloud service (S3, a Colab session's persistent/mounted storage, etc.) without checking it against the DUA first.
- **If compute moves to Redivis (see issue #23): only aggregate outputs — no per-patient rows — may ever be exported out of it.** Same rule as everywhere else in this file, extended to a new environment. The patient-similarity graph, per-patient embeddings/vectors, and per-patient predictions must stay inside Redivis; only summary tables matching the `results/aggregate/` schema (AUROC/AUPRC, homophily numbers, degree distributions, centrality stats, and figures built from those) leave. See `PROJECT_GUIDE.md`, Module H, for the full pipeline and where each stage runs.
- Nothing under `data/` other than `README.md` and `checksums.txt` may be tracked by git, ever.

Questions about access or terms: ask the team's Infra Lead before proceeding, not after.
