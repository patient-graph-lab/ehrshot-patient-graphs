# notebooks/

Colab notebooks, one per visualization/analysis goal. All computation for Project 1 happens here.

## Naming convention

`NN_short-description.ipynb`, numbered in the order they'd typically be run. Each notebook imports its actual logic from `ehrgraph` (the `src/` package) — a notebook should be mostly cells that call functions and make plots, not a second copy of the pipeline logic.

## Every notebook starts with the same bootstrap cell

```python
# Cell 1 — bootstrap (identical in every notebook in this repo)
import os
import sys
from google.colab import userdata
token = userdata.get("GH_PAT")  # set once via the key icon in Colab's left sidebar — never type a token into a cell

BRANCH = "main"  # change this to test an unmerged branch/PR before it's merged

REPO_DIR = "ehrshot-patient-graphs"
if os.path.exists(REPO_DIR):
    # Re-running in the same session without a restart — make sure we're
    # actually on BRANCH's latest commit instead of silently keeping
    # whatever was cloned before (git clone fails quietly if the dir
    # already exists, and !-shell failures don't stop the notebook).
    os.chdir(REPO_DIR)
    !git fetch origin
    !git checkout {BRANCH}
    !git reset --hard origin/{BRANCH}
else:
    !git clone --branch {BRANCH} https://{token}@github.com/patient-graph-lab/ehrshot-patient-graphs.git
    os.chdir(REPO_DIR)

!pip install -e ".[dev,data]"

# pip install -e . registers the package via a .pth file, which Python only
# reads at interpreter startup. Reinstalling into an already-running kernel
# succeeds on disk -- "Successfully installed ehrgraph-0.1.0" -- but
# "import ehrgraph" still fails until the kernel restarts, since the running
# interpreter never re-scans for new sys.path entries. Adding src/ directly
# sidesteps that: takes effect immediately, whether this is a fresh VM or a
# rerun, no restart needed.
src_path = os.path.join(os.getcwd(), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
```

**Opening the notebook from a specific branch in Colab's UI does not change what this cell clones** — that clone is a separate step, always pulls whatever `BRANCH` says, and defaults to `main` regardless of which branch you opened. Set `BRANCH` explicitly when testing an unmerged PR. And because Colab sessions often get reused across multiple "Run all"s without an actual VM restart, the cell above is written to detect an existing checkout and reset it to `BRANCH`'s latest commit rather than silently keeping stale code — don't simplify this back to a bare `git clone`.

**If `import ehrgraph` fails with `ModuleNotFoundError` later in the notebook** despite this cell reporting a successful install, it's one of two things: (1) this cell didn't actually run in the *currently active* kernel — common after a runtime reconnect, or jumping to a later cell without re-running from the top; data on disk survives a reconnect, installed-package state doesn't. Fix: re-run this cell. (2) The `sys.path` fix above wasn't in the version of this cell that ran — an older editable-install-only version of this cell needs an actual **Runtime → Restart session** after `pip install`, since Python only reads the `.pth` file at interpreter startup.

## Loading the dataset — second cell, right after the bootstrap

**Google Drive is not an option — confirmed disallowed by Stanford's data stewards, Sep 20, 2026** (general Drive doesn't meet their security requirements). Don't mount Drive or pass `source_dir` to `ensure_ehrshot_data()`; that parameter still exists in the code but must not be used — see [`DATA_ACCESS.md`](../DATA_ACCESS.md) for the full reply.

```python
from scripts.colab_setup import ensure_ehrshot_data

data_root = ensure_ehrshot_data(checksums_path="data/checksums.txt")
```

This prompts an upload dialog with no `source_dir` given — the only supported path right now. On your own machine first, `python dismantle_data.py EHRSHOT_ASSETS.zip` (grab just that one file — no repo clone needed, it has no dependencies on the rest of this package) writes `./data-chunks/` with the large file split into pieces plus copies of the other 3 (only `EHRSHOT_ASSETS.zip` is too large to upload as-is). Select everything in `data-chunks/` in the dialog; the pieces reassemble automatically. **Known cost**: measured at a fixed ~4-5 Mbps regardless of actual bandwidth — hours per session for the full dataset, not minutes.

**Stanford's actual recommendation is to skip Colab entirely and develop inside Redivis instead**, which hosts the data natively. Under evaluation, not yet decided — see `DATA_ACCESS.md` for what's confirmed and what's still open before assuming this Colab workflow is the long-term answer.

That's it — after these two cells, `from ehrgraph.graphs import ...` works, and `EHRSHOT_DATA_ROOT` is set for the rest of the session. See `00_setup_and_sanity_check.ipynb` for the working version of both.

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
