# notebooks/

Colab notebooks, one per visualization/analysis goal. All computation for Project 1 happens here.

## Naming convention

`NN_short-description.ipynb`, numbered in the order they'd typically be run. Each notebook imports its actual logic from `ehrgraph` (the `src/` package) — a notebook should be mostly cells that call functions and make plots, not a second copy of the pipeline logic.

## Every notebook starts with the same bootstrap cell

```python
# Cell 1 — bootstrap (identical in every notebook in this repo)
from google.colab import userdata
token = userdata.get("GH_PAT")  # set once via the key icon in Colab's left sidebar — never type a token into a cell
!git clone https://{token}@github.com/patient-graph-lab/ehrshot-patient-graphs.git
%cd ehrshot-patient-graphs
!pip install -e ".[dev,data]"

from scripts.colab_setup import ensure_ehrshot_data
ensure_ehrshot_data(checksums_path="data/checksums.txt")
```

That's it — after this cell, `from ehrgraph.graphs import ...` works, and `EHRSHOT_DATA_ROOT` is set for the rest of the session. See `00_setup_and_sanity_check.ipynb` for the working version of this.

## Saving results back

Any output that should be kept writes to `results/aggregate/` (never `results/per_patient/`), then:

```python
!git add results/aggregate/*.csv
!git commit -m "..."
!git push
```

## Planned notebooks (filled in as each week's work happens)

- `00_setup_and_sanity_check.ipynb` — bootstrap + confirms data loaded correctly (done)
- `01_baseline_spike.ipynb` — Week 1: CLMBR full-data logistic regression vs. Stanford's published number
- `02_graph_construction.ipynb` — Week 2: build the three graph conditions
- `03_degree_and_centrality.ipynb` — Week 3: degree distribution, the four centrality measures
- `04_community_detection.ipynb` — Week 3–4: Louvain/Leiden, Girvan-Newman dendrogram
- `05_homophily_and_energy.ipynb` — Week 3: class-insensitive homophily, normalized Dirichlet energy
- `06_disease_trajectory.ipynb` — Week 4: the trajectory graph and its dendrogram
- `07_task_b_and_comparison.ipynb` — Week 4: repeat for Task B, cross-task comparison
