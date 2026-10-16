"""One-line setup for every notebook: `from ehrgraph.project import setup; p = setup()`.

Put the professor's OMOP CSVs in `data/omop/` inside this repo (that folder
is gitignored, so they can never be committed). That's it -- no environment
variables needed. If your copy lives somewhere else, set EHRSHOT_TRIMMED_ROOT.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

REQUIRED = (
    "person.csv",
    "visit_occurrence.csv",
    "condition_occurrence.csv",
    "concept_ancestor.csv",
)
NAMES_FILE = Path(__file__).parent / "data" / "concept_names.json"


def find_repo_root() -> Path:
    here = Path.cwd()
    for path in (here, *here.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "ehrgraph").exists():
            return path
    return Path(__file__).resolve().parents[2]


@dataclass
class Project:
    root: Path
    data: Path

    @property
    def results(self) -> Path:
        return self.root / "results" / "aggregate"

    @property
    def figures(self) -> Path:
        path = self.results / "figures"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @cached_property
    def tables(self):
        from ehrgraph.data.omop import OmopTables

        return OmopTables(self.data)

    @cached_property
    def diagnoses(self):
        """Every patient's diagnoses, as standard (SNOMED) concepts."""
        from ehrgraph.data.omop import OmopDatabase

        return OmopDatabase(self.tables)

    @cached_property
    def billing_codes(self):
        """Same diagnoses, but as the original ICD-9 / ICD-10 billing codes."""
        from ehrgraph.data.omop import OmopDatabase

        return OmopDatabase(self.tables, codes="source")

    def labels(self, task: str):
        from ehrgraph.data.omop_labels import rebuild_task_labels

        return rebuild_task_labels(self.tables, task)

    def names(self, codes) -> dict:
        """'OMOP/320128' -> 'Essential hypertension'. Uses concept.csv if your
        data folder has one, otherwise the small names list bundled with the
        code (names only -- no patient data)."""
        codes = list(codes)
        if (self.data / "concept.csv").exists():
            from ehrgraph.data.omop import concept_names

            return concept_names(codes, self.data / "concept.csv")
        bundled = json.loads(NAMES_FILE.read_text()) if NAMES_FILE.exists() else {}
        return {c: bundled.get(c, c) for c in codes}


def setup() -> Project:
    root = find_repo_root()
    data = Path(os.environ.get("EHRSHOT_TRIMMED_ROOT", root / "data" / "omop"))
    missing = [name for name in REQUIRED if not (data / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Couldn't find {', '.join(missing)} in {data}.\n"
            "Put the professor's OMOP CSV files in data/omop/ inside the repo "
            "(see data/README.md), then run this cell again."
        )
    print(f"data folder: {data}\nresults go to: {root / 'results' / 'aggregate'}")
    return Project(root=root, data=data)
