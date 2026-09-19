.PHONY: graphs stats grid site test lint

graphs:
	python scripts/02_build_graphs.py

stats:
	python scripts/03_network_stats.py

grid:
	python scripts/04_run_grid.py

site:
	cd site && quarto render && quarto preview

test:
	pytest -q

lint:
	ruff check .
