"""Where each kind of data comes from (team decision, Oct 2 2026).

Two on-device data sources, and only the first is required:

  trimmed tables  The course's OMOP release (person, visit_occurrence,
                  condition_occurrence, ..., concept_ancestor) -- every team
                  member has access. PRIMARY for all Project 1 results:
                  patients, diagnoses (already SNOMED-coded) and outcome
                  labels (rebuilt with `ehrgraph.data.omop_labels`).
  full release    Stanford's EHRSHOT_ASSETS -- CLMBR fingerprints, official
                  labels/splits, published scores. OPTIONAL: only members
                  with their own approved download have it. Steps that need
                  it (baseline check, Dirichlet energy, Project 2 models)
                  are skipped cleanly when it's absent.

Nothing here copies or moves data; it only finds what is already on the
machine. Each person needs their own access to each source (DATA_ACCESS.md)
-- fingerprints and labels are patient-level and are never shared.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

TRIMMED_CANDIDATES = (
    Path.home() / "ehrshot-data" / "trimmed",
    Path.home() / "ehrshot-exploration" / "data",
)
FULL_RELEASE_DEFAULT = Path.home() / "ehrshot-data"
REQUIRED_TRIMMED = ("person.csv", "visit_occurrence.csv", "condition_occurrence.csv")


@dataclass(frozen=True)
class DataSources:
    trimmed: Path
    full: Path | None  # the directory holding EHRSHOT_ASSETS/, or None

    @property
    def has_full_release(self) -> bool:
        return self.full is not None

    @property
    def assets(self) -> Path | None:
        return self.full / "EHRSHOT_ASSETS" if self.full else None

    @property
    def clmbr_features(self) -> Path | None:
        if self.full is None:
            return None
        path = self.full / "EHRSHOT_ASSETS" / "features" / "clmbr_features.pkl"
        return path if path.exists() else None

    @property
    def concept_table(self) -> Path | None:
        """Readable concept names: the trimmed release's concept.csv if it has
        one, else the full release's copy. Vocabulary only, no patient data."""
        candidates = [self.trimmed / "concept.csv"]
        if self.full is not None:
            candidates.append(
                self.full / "EHRSHOT_ASSETS" / "femr" / "logs" / "omop_dir" / "concept.csv"
            )
        return next((p for p in candidates if p.exists()), None)

    def describe(self) -> str:
        lines = [f"trimmed tables : {self.trimmed}"]
        lines.append(
            f"full release   : {self.full}"
            if self.full
            else "full release   : not found -- CLMBR-dependent steps will be skipped"
        )
        lines.append(f"concept names  : {self.concept_table or 'not found -- codes shown as IDs'}")
        return "\n".join(lines)


def find_trimmed_root() -> Path:
    """$EHRSHOT_TRIMMED_ROOT if set, else the first conventional location
    that holds the required tables."""
    env = os.environ.get("EHRSHOT_TRIMMED_ROOT")
    candidates = [Path(env)] if env else list(TRIMMED_CANDIDATES)
    for path in candidates:
        if all((path / name).exists() for name in REQUIRED_TRIMMED):
            return path
    raise FileNotFoundError(
        "Course OMOP tables not found. Set EHRSHOT_TRIMMED_ROOT to the folder holding "
        f"{', '.join(REQUIRED_TRIMMED)} (checked: {', '.join(map(str, candidates))})."
    )


def find_full_release_root() -> Path | None:
    """$EHRSHOT_DATA_ROOT, else ~/ehrshot-data -- but only if an extracted
    EHRSHOT_ASSETS/ is actually there."""
    env = os.environ.get("EHRSHOT_DATA_ROOT")
    path = Path(env) if env else FULL_RELEASE_DEFAULT
    return path if (path / "EHRSHOT_ASSETS").is_dir() else None


def find_data_sources() -> DataSources:
    return DataSources(trimmed=find_trimmed_root(), full=find_full_release_root())
