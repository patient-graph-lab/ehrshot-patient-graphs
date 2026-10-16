# Similar Patients, Shared Outcomes

Patient similarity networks and information leakage in EHRSHOT.

Sanjay Malla, Quade Leonard, Ean Stedham. CIS 4930 / CAI 5155 Network Analysis and Machine Learning with Graphs, University of South Florida, Fall 2026.

![CI](https://github.com/patient-graph-lab/ehrshot-patient-graphs/actions/workflows/ci.yml/badge.svg?branch=trimmed-dataset)

## Overview

We turn hospital records from Stanford's EHRSHOT dataset into networks. Each node is a hospital stay, and two stays are linked when the patients' diagnosis histories are very similar. We then use classical network analysis to answer two questions:

1. **Do linked stays share outcomes?** We look at two outcomes: a hospital stay of 7 days or more, and a new diagnosis of high blood pressure within a year.
2. **What happens if the network accidentally uses information from the future?** This mistake is called *information leakage*. We compare an *honest* network (only diagnoses known at prediction time), a *cheating* network (the full record, future included) and a *random* network.

Main findings:

- Honest networks are tightly clustered, and linked stays share outcomes far more often than chance.
- Leakage depends on the question. It barely changes the long-stay network. For high blood pressure, the future record contains the diagnosis itself, so the cheating network looks better than it should.
- A disease-to-disease network recovers recognisable groups of conditions, such as heart failure, sepsis and respiratory failure.

![Honest, cheating and random networks for long hospital stays](results/aggregate/figures/02_network_pictures.png)

The full write-up is in [`report/project1.tex`](report/project1.tex). Finished notebooks with all outputs are in [`notebooks/executed/`](notebooks/executed/), and you can read them on GitHub without installing anything.

## Setup

This branch (`trimmed-dataset`) uses only the course's trimmed OMOP dataset.

**Requirements:**
- Python 3.10 or newer;
- [uv](https://docs.astral.sh/uv/);
- your own approved copy of the course dataset (`ehrshot.zip`).

**Optional:** Docker, and a LaTeX installation if you want to build the report PDF.

### 1. Install

```bash
git clone https://github.com/patient-graph-lab/ehrshot-patient-graphs.git
cd ehrshot-patient-graphs
uv sync --extra dev
uv run pre-commit install
```

### 2. Add the data

Unzip the course dataset into `data/omop/`:

```bash
unzip ~/Downloads/ehrshot.zip -x '__MACOSX/*' -d data/omop
```

The notebooks need `person.csv`, `visit_occurrence.csv`, `condition_occurrence.csv` and `concept_ancestor.csv`. The folder is gitignored, so the data cannot be committed. If you keep the data somewhere else, set `EHRSHOT_TRIMMED_ROOT` to that folder.

### 3. Run the notebooks

```bash
uv run jupyter lab
```

Open the notebooks in `notebooks/` and run them in order, `01` to `06`. Together they take about 10 minutes.

| Notebook | Contents |
|---|---|
| `01_data_and_labels` | The data, the two outcomes, and the ICD-9 to ICD-10 switch in 2015 |
| `02_long_stay_networks` | Honest, cheating and random networks for long stays |
| `03_high_blood_pressure_networks` | The same for new high blood pressure |
| `04_disease_trajectories` | Which diseases tend to follow which, and how they group |
| `05_spreading_and_partitioning` | SIR spreading, cascades, and spectral partitioning |
| `06_wrap_up` | Both outcomes side by side, plus robustness checks |

Results are written to `results/aggregate/` (summary CSVs) and `results/aggregate/figures/`.

### 4. Check that everything reproduces

```bash
uv run pytest            # unit tests on synthetic data, no dataset needed
make docker-check        # full run in a clean container
```

`make docker-check` builds a clean container, runs the tests and all six notebooks, and confirms the results match the committed ones. The data is mounted read-only, and the container has no network access. Use `DATA=/path/to/data` if your data is not in `data/omop/`, and `DOCKER=podman` to use Podman.

### 5. Build the report

```bash
make report              # writes report/project1.pdf
```

## Repository layout

```
data/omop/          course CSV files (local only, never committed)
notebooks/          analysis notebooks 01-06; executed/ holds copies with outputs
src/ehrgraph/       Python package: data loading, graph building, network measures, figures
results/aggregate/  summary CSVs and figures (no patient-level data)
report/             LaTeX report
tests/              unit tests on synthetic data
docs/               results file format and progress log
scripts/            notebook publishing and reproducibility checks
```

## Data use

The dataset is private and covered by a data use agreement:
- The repository contains only code, summary numbers and figures.
- Each team member uses their own approved copy, and the data stays on that person's computer.
- Executed notebooks are scanned for patient identifiers before they are committed, and CI checks every pull request.

See [`DATA_ACCESS.md`](DATA_ACCESS.md) for details.

## License and citation

Code: [Apache-2.0](LICENSE). Citation: [`CITATION.cff`](CITATION.cff). EHRSHOT: Wornow et al., *EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models*, NeurIPS 2023 ([arXiv:2307.02028](https://arxiv.org/abs/2307.02028)).
