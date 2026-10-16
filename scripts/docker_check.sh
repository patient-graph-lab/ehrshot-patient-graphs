#!/usr/bin/env bash
# Blind end-to-end check, run inside the Docker image (`make docker-check`):
# tests, then every notebook from scratch, then compare the fresh results
# with the committed ones. Nothing leaves the container.
set -euo pipefail
cd /app

echo "== 1/4 data folder"
python -c "from ehrgraph.project import setup; setup()"

echo "== 2/4 tests"
pytest -q

echo "== 3/4 notebooks 01-06, top to bottom"
cp -r results/aggregate /tmp/committed
for nb in notebooks/0[1-6]_*.ipynb; do
    start=$SECONDS
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
        --output-dir /tmp/executed --log-level=WARN "$nb"
    echo "   ok  $(basename "$nb")  ($((SECONDS - start))s)"
done

echo "== 4/4 fresh results vs committed results"
python scripts/compare_results.py /tmp/committed results/aggregate
