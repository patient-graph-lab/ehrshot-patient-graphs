# Contributing

Trunk-based: short-lived branches off `main`, merged via PR, squash-merged. No `develop` or `release/*` branches — there's no need for them at this scale.

## Before you start

1. `pip install -e ".[dev]"` and `pre-commit install` — every commit runs `ruff`, `nbstripout`, and a large-file guard locally, before anything reaches GitHub.
2. Get your own DUA-approved EHRSHOT access (see [`DATA_ACCESS.md`](DATA_ACCESS.md)) and set `EHRSHOT_DATA_ROOT`. You can start contributing to graph/model code against the synthetic fixtures in `tests/fixtures/` before your access comes through — no need to wait.

## Branch naming

`feat/`, `fix/`, `exp/`, `paper/`, `site/`, `docs/` prefixes. `exp/` branches may run long (an experiment can take a week) — rebase onto `main` weekly so they don't drift.

## Every PR

- Fill out the data-safety checklist in the PR template. It exists because the most common way a research project produces a wrong paper is a silent data leak or a graph-construction change that invalidates results generated earlier without anyone rerunning them.
- If you touched graph construction, time-censoring, or splits, add or update a test in `tests/test_censoring.py` or `tests/test_graphs.py`, and say whether previously-generated results are now invalidated.
- `data-guard` and `ci` must pass. `data-guard` is never bypassable, even under deadline pressure — see [`DATA_ACCESS.md`](DATA_ACCESS.md) for why.

## Tagging

Submission commits get tagged (`project1-submission`, `project2-submission`) at the exact commit whose output was submitted — costs nothing, and gives you an immutable reference if a figure looks wrong three weeks later.
