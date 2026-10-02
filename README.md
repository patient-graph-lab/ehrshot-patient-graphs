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
| See the results, with every table and figure | [`notebooks/executed/`](notebooks/executed/): GitHub shows them directly |
| Just look at the charts | [`results/aggregate/figures/`](results/aggregate/figures/) |
| Read the summary numbers | [`results/aggregate/`](results/aggregate/) (one small CSV per analysis) |
| Follow what we did, step by step | [`docs/progress-log.md`](docs/progress-log.md) |
| Understand the data rules | [`DATA_ACCESS.md`](DATA_ACCESS.md) |

<details>
<summary><b>A few words you'll see around the repo</b></summary>

- **EHRSHOT**: a public research dataset from Stanford with the anonymized medical records of about 6,700 patients.
- **CLMBR**: an AI model Stanford trained on millions of other patients. It turns each patient's history into a list of 768 numbers, a sort of "fingerprint".
- **Few-shot**: learning from very few examples (as few as 1 to 128).
- **Homophily**: how much connected patients share the same outcome. 0 means no more than chance.
- **Leakage**: future information sneaking into a prediction, i.e. the "peeking" above.
- **OMOP tables**: the standard spreadsheet-like format the medical records come in.

</details>

## Run it yourself

You'll need access to the data first; the next section explains how. Then:

**1. Install** (takes a few minutes; uses [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/patient-graph-lab/ehrshot-patient-graphs.git
cd ehrshot-patient-graphs
uv sync --extra dev
uv run pre-commit install        # safety checks that run before each commit
```

**2. Tell the code where your data is:**

```bash
export EHRSHOT_TRIMMED_ROOT=/path/to/the/course/omop/tables   # required: the folder with person.csv, visit_occurrence.csv, ...
export EHRSHOT_DATA_ROOT=/path/to/your/full/ehrshot/release   # optional: adds the CLMBR-based measures
```

**3. Open a notebook** in VS Code or Jupyter (`uv run jupyter lab`) and run it top to bottom. Start with **07** (long hospital stays) or **10** (high blood pressure). [`notebooks/README.md`](notebooks/README.md) explains what each one does and which data it needs.

**4. Check everything works** any time with `uv run pytest`. The tests use made-up data, so they run without the real dataset.

Everything runs on an ordinary laptop. Most notebooks finish in 1 to 4 minutes; the robustness checks (notebook 11) take about 10.

## About the data (please read)

The medical records come from Stanford under a data-use agreement, so **this repository contains no patient data at all**: only code, summary numbers, and charts. A few simple rules keep it that way:

- **Each person gets their own access** through Stanford's Redivis platform. Please never send data files to a teammate, even within the project.
- **The data stays on your own computer:** no Google Drive, Colab, or other cloud uploads.
- **Only summaries get committed.** Notebooks are saved without their outputs; finished copies with outputs go in `notebooks/executed/` only after a privacy check (`scripts/publish_executed.py`). Automatic checks run before every commit and on every pull request.

**Which data do you need?** Everyone on the course has the course's **OMOP tables**, and that's enough to run everything in Project 1. Stanford's **full release** adds one measure based on CLMBR fingerprints here, and it will be needed for Project 2. [`DATA_ACCESS.md`](DATA_ACCESS.md) has the details and how to request access.

## How the repo is organized

```
src/ehrgraph/       the project's Python code
  data/             reading the medical records and building outcome labels
  graphs/           turning patients into networks (honest, cheating, random)
  netstats/         measuring the networks: groups, central patients, homophily
  eval/             baseline checks, label helpers, and safety checks for anything we publish
  viz/              the charts
notebooks/          one notebook per analysis step (07-11 are the main results)
  executed/         the same notebooks with all outputs, ready to read
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

To cite this work, see [`CITATION.cff`](CITATION.cff). The code is open source under the [Apache-2.0 license](LICENSE). The data is not: it stays under Stanford's terms (see [`DATA_ACCESS.md`](DATA_ACCESS.md)).
