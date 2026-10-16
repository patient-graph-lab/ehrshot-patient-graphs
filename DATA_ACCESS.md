# Data Access

This repo has **no patient data**, not even in small pieces. Only code, summary numbers, and charts. `.gitignore`, pre-commit hooks, and a CI check (`data-guard`) enforce this on every pull request.

## What we use

- **The course dataset:** the professor's trimmed OMOP tables (a zip of CSV files). Everyone on the team uses only this.
- **EHRSHOT** (Wornow et al., NeurIPS 2023 Datasets & Benchmarks, [arXiv:2307.02028](https://arxiv.org/abs/2307.02028)): a de-identified EHR benchmark from Stanford. Our outcome definitions come from it.

## Getting your own access

1. Get your own approved access to the course data through the course's access process. Each person needs their own.
2. Never share your copy with a teammate, even someone on this project. Repo access doesn't give anyone rights to the data.
3. Put the CSV files in `data/omop/` inside the repo. That folder is gitignored.

## Where the data lives

On your own computer, and nowhere else. No Google Drive (not even temporarily), no Colab, no shared folders, no cloud uploads.

## What can leave your computer

- Code, tests, and notebooks **with outputs stripped**.
- Summary tables in `results/aggregate/` (one row per metric, never one row per patient).
- Figures built only from those summaries.
- Finished notebook copies in `notebooks/executed/`, **only** through `scripts/publish_executed.py`. That script checks every output against the real patient IDs before saving. CI checks them again.

## Small groups

Don't publish any subgroup smaller than about 20 patients. A tiny cell (for example, "3 patients, 2 positive") can identify someone. Check any demographic table by hand before it goes public.

## How a public site or paper stays safe

1. `results/aggregate/` is the only thing public pages read. It holds summaries, never one row per patient. `write_aggregate` refuses patient-identifying columns.
2. `results/per_patient/` is gitignored completely.
3. The site is built without access to the data, so it can't read patient-level data, even by accident.
4. GitHub Pages sites are public even when the repo is private. The safety comes from points 1 to 3, not from repo privacy.

Questions about access or these rules? Ask the team before going ahead, not after.
