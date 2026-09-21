"""Local-first data setup -- no Colab, no Drive, no upload widget.

The data already lives on disk (~/ehrshot-data by convention, as 4 archive
files). This extracts whichever of them haven't been extracted yet, and sets
EHRSHOT_DATA_ROOT for the rest of the session. That's the whole job.

Usage, as the first cell of any local notebook:

    from scripts.local_setup import ensure_local_data
    data_root = ensure_local_data()
"""

from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / "ehrshot-data"

#: (archive filename, marker path to check for "already extracted") -- the
#: marker is the actual directory the archive is known to produce, not a
#: separate .extracted file, so this stays correct even if only some of the
#: 4 archives are present (e.g. EHRSHOT_ASSETS.zip alone is enough for most
#: work; meds_reader_omop_ehrshot.tar.gz is only needed for issue #1-style
#: direct meds_reader access).
ARCHIVES = [
    ("EHRSHOT_ASSETS.zip", "EHRSHOT_ASSETS"),
    ("EHRSHOT_MEDS.zip", "EHRSHOT_MEDS"),
    ("meds_omop_ehrshot.tar.gz", "meds_omop_ehrshot"),
    ("meds_reader_omop_ehrshot.tar.gz", "meds_reader_omop_ehrshot"),
]


def _extract(archive_path: Path, data_dir: Path) -> None:
    print(f"Extracting {archive_path.name} (first time -- can take a few minutes)...")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(data_dir)
    else:
        with tarfile.open(archive_path) as tf:
            tf.extractall(data_dir)


def ensure_local_data(data_dir: str | Path | None = None) -> Path:
    """Extract any of the 4 archives that haven't been extracted yet, set
    EHRSHOT_DATA_ROOT, return the path.

    Only archives that are actually present in data_dir get extracted --
    doesn't require all 4 (e.g. it's fine to only have EHRSHOT_ASSETS.zip
    for most work). Raises only if NONE of the 4 are present at all.
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    found_any = False
    for archive_name, produces_dir in ARCHIVES:
        archive_path = data_dir / archive_name
        target_dir = data_dir / produces_dir
        if not archive_path.exists():
            continue
        found_any = True
        if not target_dir.exists():
            _extract(archive_path, data_dir)

    if not found_any:
        raise FileNotFoundError(
            f"None of {[a for a, _ in ARCHIVES]} found in {data_dir}. "
            "Download at least EHRSHOT_ASSETS.zip there first (see DATA_ACCESS.md) "
            "-- this script doesn't fetch data for you."
        )

    os.environ["EHRSHOT_DATA_ROOT"] = str(data_dir)
    print(f"EHRSHOT_DATA_ROOT = {data_dir}")
    return data_dir
