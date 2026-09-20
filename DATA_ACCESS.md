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

## Colab workflow (how Project 1 actually runs, for now — see "Redivis" below)

**RESOLVED, Sep 20, 2026 — Google Drive is explicitly disallowed. This is confirmed, not a judgment call.** Stanford's data stewards (shahlab-contact@stanford.edu) replied to the question sent earlier the same day: *"Generally only institution-level Google Drive setups (e.g., Stanford's) which explicitly include citable, documented guidance on protocols for storing high-risk data are permitted. The general rule is to not store the data in Google Drive, since it doesn't meet our explicit requirements around security."* This supersedes every earlier Drive-related risk analysis in this file's history (all now void) — a personal or team Google Drive, brief pass-through or persistent, does not meet Stanford's bar. **Any standing copy in Drive must be deleted immediately, Trash emptied too** (Drive's delete only moves to Trash by default).

**Current interim path: direct chunked upload, zero third-party storage of any kind.** On your own machine, `python dismantle_data.py EHRSHOT_ASSETS.zip` (pure Python, works the same on Windows/Mac/Linux — grab just that one file, no repo clone needed) writes `./data-chunks/` containing the large file split into ~400MB pieces plus copies of the other 3. Upload everything in that folder in one multi-select when `ensure_ehrshot_data(checksums_path="data/checksums.txt")` (no `source_dir`) prompts for it — the pieces reassemble automatically. **Known cost: measured at ~4.4 Mbps regardless of actual bandwidth, meaning hours per session for the full ~4.95GB dataset** (see the commit history for the original measurement) — this is the tradeoff for touching nothing but the Colab VM's own ephemeral disk.

**Stanford's actual recommendation: don't use Colab at all — develop natively within Redivis.** Their reply continued: *"The clearest advice is to do your notebook and development within Redivis, which supports Colab-style notebooks and compute and lets you house the data securely."* Initial research (Sep 20): Redivis Python notebooks run on a `jupyter/pytorch-notebook` base image (CUDA 12 / Python 3.12), support installing arbitrary pip packages via a Dependencies panel (`Packages`, `pre_install.sh`, `post_install.sh`), and offer configurable GPU/CPU/memory/disk resources — a free default tier plus custom configs for more. This looks credible as a real alternative, and would eliminate the entire data-transfer problem this section has been wrestling with, since the notebook would run where the data already lives natively.

**Not yet resolved, deliberately not decided unilaterally:**
- Whether/how a private GitHub repo (this one) can be cloned into a Redivis notebook the same way it is in Colab.
- Whether `meds_reader` (a compiled C extension) installs cleanly there.
- Actual compute quotas/cost for sustained use across a semester, and GPU availability specifics for Week 6+ GNN work.
- Migration cost: `src/ehrgraph/` and `scripts/colab_setup.py`'s core logic are plain Python, portable as-is; the notebook bootstrap cells (`google.colab`-specific: Secrets, Drive, upload widgets) would need a Redivis-native rewrite.

This needs the team to actually explore a Redivis notebook directly (you already have Redivis access, since that's how the dataset was obtained) before committing to a migration — worth doing soon, since it may remove the data-transfer problem entirely rather than continuing to work around it.

## What must never happen

- No raw data file, and nothing derived from it that could re-identify or expose patient-level records, may ever be committed to this repository or published on the project's static site — **regardless of repository visibility**. GitHub Pages sites are publicly reachable even when the source repository is private; repo privacy is not a safeguard for anything published to Pages.
- **No collaborator may upload the data to Google Drive, under any circumstances — confirmed disallowed by Stanford, Sep 20, 2026.** This is no longer a judgment call to make independently.
- No collaborator may upload the data to any other third-party cloud service (S3, a Colab session's persistent/mounted storage, etc.) without checking it against the DUA first.
- **If compute moves to Redivis (see issue #23): only aggregate outputs — no per-patient rows — may ever be exported out of it.** Same rule as everywhere else in this file, extended to a new environment. The patient-similarity graph, per-patient embeddings/vectors, and per-patient predictions must stay inside Redivis; only summary tables matching the `results/aggregate/` schema (AUROC/AUPRC, homophily numbers, degree distributions, centrality stats, and figures built from those) leave. See `PROJECT_GUIDE.md`, Module H, for the full pipeline and where each stage runs.
- Nothing under `data/` other than `README.md` and `checksums.txt` may be tracked by git, ever.

Questions about access or terms: ask the team's Infra Lead before proceeding, not after.
