.PHONY: test lint report docker-build docker-check

# Where your copy of the professor's CSVs lives (default: the repo's data/omop).
DATA ?= data/omop
DOCKER ?= docker

test:
	pytest -q

lint:
	ruff check .

# The report PDF (needs a LaTeX install). Figures come from results/aggregate/figures/.
report:
	cd report && pdflatex -interaction=nonstopmode project1.tex >/dev/null \
		&& bibtex project1 >/dev/null && pdflatex -interaction=nonstopmode project1.tex >/dev/null \
		&& pdflatex -interaction=nonstopmode project1.tex | tail -2

# Clean-machine check: build the image, then run tests + all six notebooks +
# a results comparison inside it. Data is mounted read-only; no network.
# label=disable lets SELinux hosts (Fedora) read the mount without relabeling
# your data files; it's ignored elsewhere.
docker-build:
	$(DOCKER) build -t ehrgraph-project1 .

docker-check: docker-build
	$(DOCKER) run --rm --network none --security-opt label=disable \
		-v "$(realpath $(DATA))":/app/data/omop:ro \
		ehrgraph-project1
