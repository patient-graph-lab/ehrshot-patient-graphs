# Patient Graphs: does knowing "similar patients" help an AI predict health outcomes?

![CI](https://github.com/patient-graph-lab/ehrshot-patient-graphs/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

A student research project from the University of South Florida (CIS 4930 / CAI 5155, *Network Analysis and Machine Learning with Graphs*, Fall 2026). Welcome, and thanks for stopping by. You don't need a background in medicine or machine learning to follow along.

## The idea in one minute

Hospitals would love to predict things like *"will this patient need a long stay?"* or *"will this patient develop high blood pressure in the next year?"*. The hard part is that there are often only a handful of past examples to learn from.

Our idea is the same trick a movie app uses: **if people similar to you liked a film, you probably will too.** We connect each patient to the patients with the most similar medical histories, forming a network, and ask whether those neighbors help an AI model make better predictions.

There's a catch, and it's what this project is really about. If a link between two patients was built using information from **after** the moment we're predicting, the model looks smarter than it is: it's peeking at the answer key. So we build every network three ways and compare them:

| Network | Built from | Why we build it |
|---|---|---|
| 🟦 **Honest** | Only what was known *at the time* | The version you could really use |
| 🟧 **Cheating** | The patient's *whole* record, future included | To measure how much peeking inflates results |
| 🟩 **Random** | The honest network, shuffled | To see what "no real pattern" looks like |

## What we've found so far

- **Similar patients really do tend to share outcomes,** far more than chance in every version we tested.
- **Whether cheating helps depends on the question.** For *long hospital stays*, peeking at the future barely changes the network. For *new high blood pressure*, the future record literally contains the diagnosis, and the cheating network looks noticeably better. That's exactly the trap we set out to measure.
- **Network analysis caught hidden quirks in the data,** including a split caused by the 2015 change in medical billing codes, before any AI model was trained.

Project 1 (network analysis) is due Oct 23, 2026. Project 2, which tests whether the honest networks actually improve predictions, is due Dec 4, 2026.

## Have a look around (no setup needed)

| I want to... | Go here |
|---|---|
| See the results, with every table and figure | [`notebooks/executed/`](notebooks/executed/): finished notebooks, GitHub shows them directly |
| Just look at the charts | [`results/aggregate/figures/`](results/aggregate/figures/) |
| Read the summary numbers | [`results/aggregate/`](results/aggregate/) (one small CSV per analysis) |
| Read the report draft | [`report/`](report/) (LaTeX) |
| Walk through the analysis | [`notebooks/`](notebooks/): six notebooks, `01` to `06`. Start with `01` |
| Follow what we did, step by step | [`docs/progress-log.md`](docs/progress-log.md) |
| Understand the data rules | [`DATA_ACCESS.md`](DATA_ACCESS.md) |

<details>
<summary><b>A few words you'll see around the repo</b></summary>

- **EHRSHOT**: a public research dataset from Stanford with the anonymized medical records of about 6,700 patients. Our outcome definitions come from it.
- **Few-shot**: learning from very few examples (as few as 1 to 128).
- **Homophily**: how much connected patients share the same outcome. 0 means no more than chance.
- **Leakage**: future information sneaking into a prediction, i.e. the "peeking" above.
- **OMOP tables**: the standard spreadsheet-like format the medical records come in.

</details>

## Run it yourself

You'll need the course data first (see the next section). Then:

**1. Install.** You need [uv](https://docs.astral.sh/uv/). Then:

```bash
git clone https://github.com/patient-graph-lab/ehrshot-patient-graphs.git
cd ehrshot-patient-graphs
uv sync --extra dev
uv run pre-commit install        # safety checks that run before each commit
```

**2. Put the data in place.** Unzip the professor's dataset straight into `data/omop/`:

```bash
unzip ~/Downloads/ehrshot.zip -d data/omop   # use wherever you saved the zip
```

That's the whole data setup. That folder is gitignored, so the data can never get committed. You need at least `person.csv`, `visit_occurrence.csv`, `condition_occurrence.csv`, and `concept_ancestor.csv`. The other course CSVs can sit there too.

**3. Open a notebook** (`uv run jupyter lab`, or VS Code) and run it top to bottom. Start with **01**. Every notebook begins with these two lines, and they find the data for you:

```python
from ehrgraph.project import setup
p = setup()
```

If your data lives somewhere else, set `EHRSHOT_TRIMMED_ROOT` to that folder first.

**4. Check everything works** any time with `uv run pytest`. The tests use made-up data, so they run without the real dataset.

Most notebooks finish in a few minutes on an ordinary laptop.

**Or let Docker do all of it.** With [Docker](https://docs.docker.com/get-docker/) installed and the data in `data/omop/`, run `make docker-check`. This builds a clean machine with the exact locked packages, runs the tests, runs notebooks 01 to 06 from scratch, and checks that the fresh results match the committed ones. The data is mounted read-only, never copied into the image, and the container has no network. Your data lives somewhere else? Use `make docker-check DATA=/path/to/folder`. Using Podman? Add `DOCKER=podman`.

Don't want to run anything? Read the finished notebooks in [`notebooks/executed/`](notebooks/executed/) and the charts in [`results/aggregate/figures/`](results/aggregate/figures/).

## About the data (please read)

The data is the professor's course dataset, and it is private. So **this repository contains no patient data at all**: only code, summary numbers, and charts. Three rules keep it that way:

- **No patient data in the repo, ever.** Only code, summary numbers, and charts get committed.
- **Get your own access, and never send data files to anyone.** Each person requests their own approved access. That includes teammates.
- **The data stays on your own computer.** No Google Drive, Colab, or other cloud uploads.

Notebooks are saved without their outputs, and finished copies are privacy-checked before they go on GitHub. Full rules are in [`DATA_ACCESS.md`](DATA_ACCESS.md).

Stanford's full release and its CLMBR measures are not part of Project 1. Project 2 may revisit them.

## How the repo is organized

```
data/omop/          the professor's CSV files go here (gitignored, never committed)
notebooks/          the six analysis notebooks, 01 to 06
  executed/         the same notebooks with all outputs, ready to read
src/ehrgraph/       the project's Python code
  project.py        setup(): finds the data for you
  data/             reading the medical records and building outcome labels
  graphs/           turning patients into networks (honest, cheating, random)
  netstats/         measuring the networks: groups, central patients, homophily
  eval/             baseline checks, label helpers, and safety checks for anything we publish
  viz/              the charts
report/             the LaTeX draft of the report
results/aggregate/  summary numbers (CSV) and figures; the only results we publish
tests/              automated tests, all on made-up data
docs/               the progress log and the results file format
scripts/            setup and publishing helpers
r-viz/              optional R scripts for report figures
```

## Working on the project

- Make a short-lived branch, open a pull request, and let the checks run.
- If you change how networks, time cut-offs, or labels are built, add a test and say in the pull request whether earlier results change.
- Questions are welcome. Open an issue or ask the team.

## Citing and license

To cite this work, see [`CITATION.cff`](CITATION.cff). The code is open source under the [Apache-2.0 license](LICENSE). The data is not: it stays under the course's terms (see [`DATA_ACCESS.md`](DATA_ACCESS.md)).
