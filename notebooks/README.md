# Notebooks

Six notebooks, meant to be run in order. They run on your own computer and read the course data from `data/omop/`.

| # | Notebook | What it does | Run time |
|---|---|---|---|
| 01 | `01_data_and_labels` | What's in the tables, the two questions, how the labels are built, the no-peeking rule, and the 2015 ICD-9 to ICD-10 code switch | ~1 min |
| 02 | `02_long_stay_networks` | Honest, cheating, and random networks for long hospital stays | ~3 min |
| 03 | `03_high_blood_pressure_networks` | The same three networks for new high blood pressure | ~2 min |
| 04 | `04_disease_trajectories` | Which diseases follow which within a year, a Girvan-Newman family tree, and why billing codes split in 2015 | ~1 min |
| 05 | `05_spreading_and_partitioning` | An SIR epidemic model and cascades on the networks, plus spectral partitioning vs. Leiden | ~2 min |
| 06 | `06_wrap_up` | Both questions side by side, plus a check that the 2015 code switch isn't behind the results | ~2 min |

## How to run

Run `uv run jupyter lab` from the repo root (or open the notebook in VS Code), then run it top to bottom. Every notebook starts with the same two lines:

```python
from ehrgraph.project import setup

p = setup()
```

`setup()` finds the data in `data/omop/` automatically. If your data lives somewhere else, set `EHRSHOT_TRIMMED_ROOT` to that folder first.

## Where results go

Summary CSVs go to `results/aggregate/` and figures go to `results/aggregate/figures/`. Never patient-level rows. Notebooks save results with `write_aggregate` from `ehrgraph.eval.aggregate`, which refuses patient-identifying columns.

## Publishing a finished copy

Notebooks in this folder are saved without outputs. To share a finished copy with outputs, run:

```bash
uv run python scripts/publish_executed.py notebooks/02_long_stay_networks.ipynb
```

That checks every output against the real patient IDs first, and only then saves the copy to `notebooks/executed/`.
