"""Publishing executed notebooks (with outputs) safely -- team request, Oct 2 2026.

Source notebooks in notebooks/ stay output-stripped (nbstripout). Executed
copies, outputs included, live in notebooks/executed/ so team members
without CLMBR access or the compute can read every result without running
anything. That relaxes one guard -- printed outputs can leak patient-level
data -- so every executed copy must pass `scan_notebook` before it is
written there, at two strengths:

  local (scripts/publish_executed.py)  checks outputs against the REAL
      patient ids and prediction times on this machine -- exact, but only
      possible where the data is.
  CI (scripts/check_executed_notebooks.py)  data-free heuristics on every
      pull request: no id-shaped numbers, no patient columns, no long
      tables, no prediction-time timestamps, size cap.

Images are not scanned: figures are built from aggregate numbers or are
anonymous layouts (results/aggregate/figures/ holds the same PNGs).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

FORBIDDEN_COLUMNS = ("patient_id", "person_id", "subject_id", "omop_person_id", "prediction_time")
MAX_TABLE_ROWS = 100
MAX_BYTES = 5_000_000
#: A bare 9-10 digit integer -- the shape of an EHRSHOT person id -- unless it
#: is clearly a vocabulary concept id ("OMOP/320128", "concept 40511000").
_ID_SHAPED = re.compile(r"(?<![\w/.])(?<!concept )\d{9,10}(?![\w.])")
#: "2016-01-01 23:59" -- prediction times are minute-resolution timestamps.
_TIMESTAMP = re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}")
_ROWS = re.compile(r"\[(\d+) rows x \d+ columns\]")


def output_texts(nb: dict) -> list[tuple[int, str]]:
    """(cell index, text) for every textual output: streams, plain-text and
    HTML reprs, and error tracebacks."""
    found = []
    for i, cell in enumerate(nb.get("cells", [])):
        for out in cell.get("outputs", []):
            parts = []
            if "text" in out:
                parts.append(out["text"])
            data = out.get("data", {})
            for key in ("text/plain", "text/html", "text/markdown"):
                if key in data:
                    parts.append(data[key])
            if "traceback" in out:
                parts.append(out["traceback"])
            for part in parts:
                found.append((i, "".join(part) if isinstance(part, list) else str(part)))
    return found


def scan_notebook(
    nb: dict, sensitive: set[str] | None = None, size: int | None = None
) -> list[str]:
    """Problems that block publishing; empty list = safe to publish.
    `sensitive`: exact strings that must never appear (real patient ids,
    prediction times) -- pass it whenever the data is on this machine."""
    problems = []
    if size is not None and size > MAX_BYTES:
        problems.append(f"notebook is {size:,} bytes (limit {MAX_BYTES:,})")
    for i, text in output_texts(nb):
        lowered = text.lower()
        for col in FORBIDDEN_COLUMNS:
            if re.search(rf"\b{col}\b", lowered):
                problems.append(f"cell {i}: output mentions column '{col}'")
        for match in _ROWS.finditer(text):
            if int(match.group(1)) > MAX_TABLE_ROWS:
                problems.append(f"cell {i}: a {match.group(1)}-row table (limit {MAX_TABLE_ROWS})")
        if _TIMESTAMP.search(text):
            problems.append(f"cell {i}: minute-resolution timestamp (prediction-time shaped)")
        if _ID_SHAPED.search(text):
            problems.append(f"cell {i}: id-shaped number {_ID_SHAPED.search(text).group(0)}")
        if sensitive:
            hits = [s for s in sensitive if s in text]
            if hits:
                problems.append(f"cell {i}: contains {len(hits)} real patient id(s)/time(s)")
    return problems


def sanitize(nb: dict, home: Path | None = None) -> dict:
    """Copy of `nb` with this machine's home path replaced by "~" and
    per-run execution metadata removed, so republishing an unchanged
    notebook doesn't produce a noisy diff."""
    nb = copy.deepcopy(nb)
    home_str = str(home or Path.home())
    for cell in nb.get("cells", []):
        cell.get("metadata", {}).pop("execution", None)
        for out in cell.get("outputs", []):
            if "text" in out:
                out["text"] = _replace(out["text"], home_str)
            for key in ("text/plain", "text/html"):
                if key in out.get("data", {}):
                    out["data"][key] = _replace(out["data"][key], home_str)
    return nb


def _replace(value, home: str):
    if isinstance(value, list):
        return [v.replace(home, "~") for v in value]
    return value.replace(home, "~")
