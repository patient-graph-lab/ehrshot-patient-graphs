.PHONY: test lint

test:
	pytest -q

lint:
	ruff check .

# More targets (graphs, stats, grid, site) get added here as the corresponding
# scripts/ and site/ directories are created — see README for why they don't
# exist yet.
