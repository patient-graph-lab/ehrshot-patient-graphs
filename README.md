# Leakage-Safe Patient Graphs for Few-Shot Clinical Prediction

![CI](https://github.com/patient-graph-lab/ehrshot-patient-graphs/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Data](https://img.shields.io/badge/data-DUA%20required-critical)

> **Status: in progress.** This README will carry the project's actual finding once results exist. For now: does a leakage-safe patient-similarity graph genuinely improve a pretrained clinical foundation model's (CLMBR) few-shot predictions on EHRSHOT, and does the size of that improvement track graph homophily — or does it only appear when the graph is allowed to see the future?

## Data access — read this before installing anything

This repository contains **no patient data of any kind**, and never will. EHRSHOT is distributed by Stanford under a Data Use Agreement; each collaborator must obtain their own approved access. See [`DATA_ACCESS.md`](DATA_ACCESS.md) for the exact steps, and set `EHRSHOT_DATA_ROOT` to point at your own local copy before running anything here.

## TL;DR

*(Filled in once results exist — see [`docs/analysis-plan.md`](docs/analysis-plan.md) for the pre-registered primary endpoint, once it's written.)*

## Install

```bash
git clone https://github.com/patient-graph-lab/ehrshot-patient-graphs.git
cd ehrshot-patient-graphs
pip install -e ".[dev]"
pre-commit install
export EHRSHOT_DATA_ROOT=/path/to/your/local/ehrshot/copy
```

## Reproduce

```bash
make graphs   # build the three graph conditions
make stats    # network analysis (Project 1)
make grid     # run the full model grid (Project 2)
make site     # render the Quarto site locally
```

See [`Makefile`](Makefile) for what each target actually runs and its rough wall-clock time, once these are implemented.

## Repo map

```
src/ehrgraph/        installable package — data loading, graph construction, network stats, models, eval, viz
scripts/             thin CLI entry points, one per pipeline stage
configs/             experiment configs (YAML)
tests/                leakage-safety and correctness tests, run against synthetic fixtures only
paper/               Quarto sources for the two course reports and the paper
site/                Quarto site — published via GitHub Pages
docs/adr/            architecture / analysis decision records
results/aggregate/   committed — metrics only, never patient-level
results/per_patient/ gitignored — never committed
```

## Results

*(Headline table + link to the site, once results exist.)*

## How to cite

See [`CITATION.cff`](CITATION.cff).

## License and data terms

Code is licensed under [Apache-2.0](LICENSE). This license covers code only — see [`DATA_ACCESS.md`](DATA_ACCESS.md) for data terms, which are separate and non-negotiable.

## Team

CIS 4930 / CAI 5155, Network Analysis and Machine Learning with Graphs, University of South Florida, Fall 2026.
