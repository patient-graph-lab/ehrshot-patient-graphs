.PHONY: test lint report overleaf docker-build docker-check

# Where your copy of the professor's CSVs lives (default: the repo's data/omop).
DATA ?= data/omop
DOCKER ?= docker

test:
	pytest -q

lint:
	ruff check .

# The report PDF (needs a LaTeX install): report/main.pdf
report:
	cd report && pdflatex -interaction=nonstopmode main.tex >/dev/null \
		&& bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null \
		&& pdflatex -interaction=nonstopmode main.tex | tail -2

# A zip of the report with its figures, ready for Overleaf's
# "New Project > Upload Project": report/overleaf.zip
overleaf:
	rm -rf report/figures report/overleaf.zip
	mkdir -p report/figures
	cp results/aggregate/figures/*.png report/figures/
	cd report && zip -qr overleaf.zip main.tex references.bib sections figures
	rm -rf report/figures
	@echo "wrote report/overleaf.zip"

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
