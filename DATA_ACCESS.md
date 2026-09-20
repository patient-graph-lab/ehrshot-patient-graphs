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

## Colab workflow (how Project 1 actually runs)

Computation happens in Google Colab. The dataset lives on Colab's ephemeral local disk (`~/EHRSHOT`) every session and is never committed — `scripts/colab_setup.py` handles getting it there and always resolves to that same path for consistency across notebooks and sessions.

**How the data actually gets onto the VM: a brief Google Drive pass-through, not the browser-upload widget.** The original plan was to re-upload directly via `google.colab.files.upload()` each session with no Drive involved at all, specifically to avoid the open question below about whether the DUA permits a standing copy on a third-party cloud service. In practice, that widget can't handle a 4.3GB file — it routes data through a base64-encoded message channel built for small files, not a real streaming upload, and stalls indefinitely on something this large (confirmed Sep 20: 8+ minutes at 0% with no RAM movement).

The working alternative: upload the archive files to a Drive folder once (Drive's own uploader is a real resumable upload, so this is fast and reliable), then `ensure_ehrshot_data(source_dir="/content/drive/MyDrive/ehrshot_transfer")` copies them from Drive to the VM's local disk — call `delete_from_drive(...)` immediately after, which the setup notebook does automatically. **This means the data does touch Drive, for a few minutes, even though nothing persists there afterward.**

**Reviewed against the actual EHRSHOT Data Set License (Sep 20, 2026).** The license doesn't mention cloud storage at all — it neither authorizes nor forbids it. Two clauses are relevant: Clause 3 ("will not share access... with anyone else") is about sharing with other *people*, not storage infrastructure, so a private, unshared Drive folder doesn't trigger it. Clause 4 ("reasonable and prudent care to maintain physical and electronic security") is the actual operative, judgment-based standard, and the license gives no bright line for it. The team's decision: a private, never-link-shared, immediately-deleted Drive folder under the license holder's own account is acceptable under that standard, given the data is de-identified (Clause 5) and the exposure window is minutes. **This is the team's own risk read, not a legal guarantee or something Stanford confirmed** — the license holder carries the indemnification obligation (the license's own terms), so this was a deliberate call made with the actual license text in hand, not an assumption. If anyone wants certainty instead of a reasonable reading, email shahlab-contact@stanford.edu — that offer stands regardless of this decision.

Cloning this **private** repo from within Colab needs a GitHub token: generate a fine-grained personal access token scoped to just this repo, and store it in Colab's Secrets panel (the key icon in the left sidebar, retrieved via `google.colab.userdata.get(...)`) — never typed into a cell in plaintext, where it could leak if the notebook is ever shared.

## What must never happen

- No raw data file, and nothing derived from it that could re-identify or expose patient-level records, may ever be committed to this repository or published on the project's static site — **regardless of repository visibility**. GitHub Pages sites are publicly reachable even when the source repository is private; repo privacy is not a safeguard for anything published to Pages.
- No collaborator may upload the data to a third-party cloud service (Google Drive, S3, a Colab session's persistent/mounted storage, etc.) without checking it against the DUA first. A Colab notebook that mounts Drive is a common accidental-redistribution path — be deliberate about it.
- Nothing under `data/` other than `README.md` and `checksums.txt` may be tracked by git, ever.

Questions about access or terms: ask the team's Infra Lead before proceeding, not after.
