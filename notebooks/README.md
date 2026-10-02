# notebooks/

One notebook per analysis goal, run locally. All computation for Project 1 happens here.

## Naming convention

`NN_short-description.ipynb`, numbered in the order they'd typically be run. Each notebook imports its actual logic from `ehrgraph` (the `src/` package) — a notebook should be mostly cells that call functions and make plots, not a second copy of the pipeline logic.

## Every notebook starts with the same bootstrap cell

```python
# Cell 1 — bootstrap (identical in every notebook in this repo)
import sys
from pathlib import Path

_repo_root = Path.cwd()
while not (_repo_root / "pyproject.toml").exists() and _repo_root != _repo_root.parent:
    _repo_root = _repo_root.parent
sys.path.insert(0, str(_repo_root))

from scripts.local_setup import ensure_local_data  # noqa: E402

data_root = ensure_local_data()
```

That's it — after this cell, `from ehrgraph.graphs import ...` works, and `EHRSHOT_DATA_ROOT` is set. No Colab, no Drive, no upload widget, no token. `ensure_local_data()` extracts whichever of the 4 archives in `~/ehrshot-data/` haven't been extracted yet and is safe to call every time (skips anything already done).

**Why the `sys.path` walk-up instead of a bare `from scripts.local_setup import ...`:** `scripts/` isn't an installed package the way `ehrgraph` is (via `uv sync`), so it's only importable if the repo root is on `sys.path`. The walk-up finds that root regardless of whether Jupyter launched from `notebooks/`, the repo root, or VS Code's own default working directory — don't simplify this to an `os.chdir()` instead, since that would break any cell using a path relative to wherever the notebook file itself lives.

## No Colab, no uploads — data stays on this device

Per `DATA_ACCESS.md` (team decision, Sep 25), notebooks run locally only. `scripts/colab_setup.py` still exists and is still tested, but it is not to be used while the device-only rule stands. Outputs are stripped by `nbstripout` before every commit (pre-commit + CI both enforce it) — printed cells can contain patient IDs, so the *only* results that reach GitHub are the files a notebook deliberately writes to `results/aggregate/`.

## Saving results back

Any output that should be kept writes to `results/aggregate/` (never `results/per_patient/`) via `ehrgraph.eval.aggregate.write_aggregate`, which enforces the long-format schema (`task, condition, graph_variant, metric, value, ci_low, ci_high, seed` — frozen for good in issue #18) and refuses patient-identifying columns. Figures go to `results/aggregate/figures/`, built only from those summaries. Then a normal commit:

```bash
git add results/aggregate/
git commit -m "..."
git push
```

## Notebook plan

| # | Notebook | Issue(s) | Data source |
|---|---|---|---|
| 00 | `setup_and_sanity_check` | #1 | full dataset (`meds_reader`) |
| 01 | `baseline_spike` | #2 | full dataset (`clmbr_features.pkl`, official labels/splits) |
| 02 | `official_labels_and_splits` | #3 | full dataset (benchmark labels for both locked tasks) |
| 03 | `patient_vectorization` | #4 | trimmed OMOP tables (`condition_occurrence`, `drug_exposure`, `procedure_occurrence`) |
| 04 | `similarity_graph_construction` | #5 | vectors from 03 |
| 05 | `leakage_conditions` | #6, #7, #8 | trimmed tables (content) + full dataset (`prediction_time` for censoring) |
| — | *(no notebook — `tests/test_censoring.py`)* | #9 | synthetic fixture only |
| 06 | `disease_trajectory_graph` | #10 | trimmed OMOP tables (`condition_occurrence` self-join) |
| 07 | `network_stats_task_a` | #11, #12, #13, #15 | full dataset, via `ehrgraph.netstats.task_analysis` |
| 08 | `girvan_newman_dendrogram` | #14 | graph from 06 |
| 09 | `cross_task_comparison` | #17 | **aggregate CSVs only** -- needs no patient data |
| 10 | `network_stats_task_b` | #16, #15 | full dataset, same library code as 07 |

Once 07/10 have written real output to `results/aggregate/`, `r-viz/figures.R` takes over for the final report figures (`r-viz/README.md`).
