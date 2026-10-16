#!/usr/bin/env bash
#
# One-step setup for "Similar Patients, Shared Outcomes".
#
# What it does, in order:
#   1. Checks the tools we need (uv, unzip; Docker and LaTeX only if asked).
#   2. Creates the Python environment with uv, from the locked versions in uv.lock.
#   3. Installs the pre-commit safety checks (they stop data from being committed).
#   4. Puts the course dataset in data/omop/ (unzipping it if you give the zip).
#   5. Runs the unit tests.
#   6. Optionally: runs the full Docker check and/or builds the report PDF.
#
# Usage:
#   ./setup.sh                                 # environment only (data already in data/omop/)
#   ./setup.sh ~/Downloads/ehrshot.zip         # also unzip the course data
#   ./setup.sh ~/Downloads/ehrshot.zip --docker --report
#
# Flags:
#   --docker   after setup, run `make docker-check` (all notebooks in a clean
#              container, about 10 minutes, compares results with the repo).
#              Uses Docker, or Podman if Docker isn't installed; set
#              DOCKER=podman to choose explicitly.
#   --report   after setup, build report/main.pdf (needs pdflatex + bibtex)
#
# Safe to run more than once: every step skips work that is already done.

set -euo pipefail   # stop on the first error, on unset variables, and on failed pipes

cd "$(dirname "$0")"   # always run from the repo root, wherever it was called from

# The four course tables the notebooks need.
REQUIRED=(person.csv visit_occurrence.csv condition_occurrence.csv concept_ancestor.csv)
DATA_DIR="data/omop"

ZIP=""
RUN_DOCKER=false
BUILD_REPORT=false
for arg in "$@"; do
    case "$arg" in
        --docker) RUN_DOCKER=true ;;
        --report) BUILD_REPORT=true ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) ZIP="$arg" ;;
    esac
done

step() { printf '\n==> %s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- 1. Tools -----------------------------------------------------------------
step "Checking tools"
command -v uv >/dev/null \
    || fail "uv is not installed. Install it with:
       curl -LsSf https://astral.sh/uv/install.sh | sh
   then open a new terminal and run this script again."
echo "uv: $(uv --version)"
if [[ -n "$ZIP" ]]; then
    command -v unzip >/dev/null || fail "unzip is not installed (e.g. 'sudo dnf install unzip' or 'sudo apt install unzip')."
fi
if $RUN_DOCKER; then
    # Use $DOCKER if set (e.g. DOCKER=podman), otherwise Docker, otherwise Podman.
    ENGINE="${DOCKER:-}"
    if [[ -z "$ENGINE" ]]; then
        if command -v docker >/dev/null; then ENGINE=docker
        elif command -v podman >/dev/null; then ENGINE=podman
        else fail "--docker was given but neither Docker nor Podman is installed (https://docs.docker.com/get-docker/)."
        fi
    fi
    # `info` fails if the daemon isn't running or you lack permission to use it.
    "$ENGINE" info >/dev/null 2>&1 \
        || fail "$ENGINE is installed but not usable. Start it (e.g. 'sudo systemctl start docker') or add yourself to the docker group, then try again."
    echo "container engine: $ENGINE"
fi
if $BUILD_REPORT; then
    command -v pdflatex >/dev/null || fail "--report was given but pdflatex is not installed (install TeX Live, or use Overleaf: make overleaf)."
fi

# --- 2. Python environment ----------------------------------------------------
# `--frozen` installs exactly what uv.lock says, so everyone gets the same
# package versions. uv downloads a matching Python itself if needed.
step "Creating the Python environment (.venv)"
uv sync --frozen --extra dev

# --- 3. Safety checks before each commit -------------------------------------
# Strips notebook outputs, blocks data-shaped files, scans executed notebooks.
step "Installing pre-commit checks"
if [[ -d .git ]]; then
    uv run --extra dev pre-commit install
else
    echo "Not a git checkout; skipping."
fi

# --- 4. Data ------------------------------------------------------------------
# data/ is gitignored, so nothing placed here can be committed.
step "Checking the course data in $DATA_DIR/"
mkdir -p "$DATA_DIR"
if [[ -n "$ZIP" ]]; then
    [[ -f "$ZIP" ]] || fail "No such file: $ZIP"
    echo "Unzipping $ZIP (this can take a minute; files already there are kept)"
    # -n: never overwrite existing files. -x: skip macOS metadata folders.
    unzip -n -q "$ZIP" -x '__MACOSX/*' -d "$DATA_DIR"
fi
missing=()
for f in "${REQUIRED[@]}"; do
    [[ -f "$DATA_DIR/$f" ]] || missing+=("$f")
done
if (( ${#missing[@]} )); then
    echo "Missing in $DATA_DIR/: ${missing[*]}"
    echo "Run again with the path to the course zip, e.g.: ./setup.sh ~/Downloads/ehrshot.zip"
    DATA_READY=false
else
    echo "All ${#REQUIRED[@]} required tables found."
    DATA_READY=true
fi

# --- 5. Unit tests --------------------------------------------------------------
# These use small made-up data, so they pass even without the real dataset.
step "Running unit tests"
uv run --extra dev pytest -q

# --- 6. Optional extras -------------------------------------------------------
if $RUN_DOCKER; then
    $DATA_READY || fail "--docker needs the data in $DATA_DIR/ first."
    # Builds a clean image (no data inside), mounts the data read-only, runs
    # with no network, executes notebooks 01-06, and compares the results.
    step "Running the full Docker check (about 10 minutes)"
    make docker-check DOCKER="$ENGINE"
fi

if $BUILD_REPORT; then
    step "Building the report"
    make report
    echo "Report: report/main.pdf"
fi

# --- Done -----------------------------------------------------------------------
step "Setup complete"
if $DATA_READY; then
    echo "Next: uv run jupyter lab   then run notebooks/01 to 06 in order."
else
    echo "Add the data (see above), then: uv run jupyter lab"
fi
