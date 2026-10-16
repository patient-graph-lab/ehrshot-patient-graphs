# Data access

This repository contains no patient data: only code, summary numbers and figures. `.gitignore`, pre-commit hooks and the `data-guard` CI check enforce this.

## The dataset

We use the course's trimmed OMOP tables, a subset of Stanford's EHRSHOT dataset (Wornow et al., NeurIPS 2023, [arXiv:2307.02028](https://arxiv.org/abs/2307.02028)). Our outcome definitions follow the EHRSHOT benchmark.

## Rules

1. **Get your own access.** Each team member needs their own approved copy through the course. Repository access does not give anyone rights to the data.
2. **Never share data files**, not even with teammates.
3. **Keep the data on your own computer**, in `data/omop/` (gitignored). No cloud drives, Colab or shared folders.
4. **Publish summaries only.** Only these may be committed:
   - code;
   - notebooks with outputs stripped;
   - summary tables in `results/aggregate/` (one row per measurement, never one row per patient);
   - figures made from those summaries.
5. **No small groups.** Don't publish any group of fewer than about 20 patients.

## Executed notebooks

Copies with outputs live in `notebooks/executed/`. They are added only through `scripts/publish_executed.py`. That script checks every output against the real patient IDs on your machine and removes local paths before saving. CI runs the same scan again on every pull request.

Questions about these rules? Ask the team before going ahead.
