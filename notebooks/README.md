# notebooks/

One notebook per analysis goal, run locally. All computation for Project 1 happens here.

## Naming convention

`NN_short-description.ipynb`, numbered in the order they'd typically be run. Each notebook imports its actual logic from `ehrgraph` (the `src/` package) — a notebook should be mostly cells that call functions and make plots, not a second copy of the pipeline logic.

## Just want the results? Read `notebooks/executed/`

Executed copies of the results notebooks (07–11), **outputs included**, live in `notebooks/executed/`. GitHub renders them directly, so you can read every table and figure without data, CLMBR access or compute. They are published with `scripts/publish_executed.py`, which privacy-checks every output against the real patient IDs on the publishing machine; CI re-checks them without data. The notebooks in this folder are the clean source versions (outputs stripped).

## Which data each notebook needs

- **Course OMOP tables (everyone has these):** notebooks 07, 08, 10 and 11. They start with:

```python
import sys
from pathlib import Path

_repo_root = Path.cwd()
while not (_repo_root / "pyproject.toml").exists() and _repo_root != _repo_root.parent:
    _repo_root = _repo_root.parent
sys.path.insert(0, str(_repo_root))

from scripts.local_setup import ensure_data_sources  # noqa: E402

sources = ensure_data_sources()  # course tables required; full release optional
```

`ensure_data_sources()` finds the course tables (`$EHRSHOT_TRIMMED_ROOT`, else `~/ehrshot-data/trimmed`, else `~/ehrshot-exploration/data`). It also finds Stanford's full release if *you* have it (`$EHRSHOT_DATA_ROOT`, default `~/ehrshot-data`). Without the full release, the only things skipped are CLMBR-based steps: Dirichlet energy here, and the official-label comparison in 11.

- **Stanford's full release (your own download):** notebooks 00–06 and 01 (the baseline check). They are the project's development history and start with `ensure_local_data()` instead. Their results were folded into 07–11, so you don't need to run them.
- **Neither:** notebook 09 reads only `results/aggregate/*.csv`.

**Why the `sys.path` walk-up instead of a bare `from scripts.local_setup import ...`:** `scripts/` isn't an installed package the way `ehrgraph` is (via `uv sync`), so it's only importable if the repo root is on `sys.path`. The walk-up finds that root regardless of whether Jupyter launched from `notebooks/`, the repo root, or VS Code's own default working directory — don't simplify this to an `os.chdir()` instead, since that would break any cell using a path relative to wherever the notebook file itself lives.

## No Colab, no uploads — data stays on this device

Per `DATA_ACCESS.md`, notebooks run locally only. Source notebooks have their outputs stripped by `nbstripout` before every commit (pre-commit and CI both enforce it), because printed cells can contain patient IDs. Results reach GitHub only as `results/aggregate/` files and as privacy-checked executed copies in `notebooks/executed/`.

## Saving results back

Any output that should be kept writes to `results/aggregate/` (never `results/per_patient/`) via `ehrgraph.eval.aggregate.write_aggregate`, which enforces the long-format schema (`task, condition, graph_variant, metric, value, ci_low, ci_high, seed` — frozen at version 1.0, see `docs/results-schema.md`) and refuses patient-identifying columns. Figures go to `results/aggregate/figures/`, built only from those summaries. Then a normal commit:

```bash
git add results/aggregate/
uv run python scripts/publish_executed.py notebooks/07_network_stats_task_a.ipynb  # optional: refresh the executed copy
git add notebooks/executed/
git commit -m "..."
```

## Notebook plan

| # | Notebook | Issue(s) | Data source |
|---|---|---|---|
| 00 | `setup_and_sanity_check` | #1 | full release (`meds_reader`): development history |
| 01 | `baseline_spike` | #2 | full release (`clmbr_features.pkl`, official labels/splits) |
| 02 | `official_labels_and_splits` | #3 | full release: development history |
| 03 | `patient_vectorization` | #4 | full release (`meds_reader`): development history |
| 04 | `similarity_graph_construction` | #5 | full release: development history |
| 05 | `leakage_conditions` | #6, #7, #8 | full release: development history |
| — | *(no notebook — `tests/test_censoring.py`)* | #9 | synthetic fixture only |
| 06 | `disease_trajectory_graph` | #10 | full release: development history (superseded by 08) |
| 07 | `network_stats_task_a` | #11–#13, #15 | **course OMOP tables**; CLMBR optional |
| 08 | `girvan_newman_dendrogram` | #14 | **course OMOP tables** (billing codes and SNOMED concepts) |
| 09 | `cross_task_comparison` | #17 | **aggregate CSVs only**: needs no patient data |
| 10 | `network_stats_task_b` | #16, #15 | **course OMOP tables**; CLMBR optional |
| 11 | `robustness_checks` | #35, #37 | **course OMOP tables**; full release optional (official-label comparison) |

Once 07/10 have written real output to `results/aggregate/`, `r-viz/figures.R` takes over for the final report figures (`r-viz/README.md`).
