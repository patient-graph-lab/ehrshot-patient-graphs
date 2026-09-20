"""Bootstrap EHRSHOT data access for a Colab session.

Convention for this project: data always lives at ~/EHRSHOT, re-uploaded fresh
every session. Nothing about the dataset is ever persisted to Drive or committed
to git — see DATA_ACCESS.md for why.

Usage, as the first real cell of any notebook (after cloning the repo and
running `pip install -e ".[dev]"`):

    from scripts.colab_setup import ensure_ehrshot_data
    ensure_ehrshot_data(checksums_path="data/checksums.txt")
"""

from __future__ import annotations

import hashlib
import os
import tarfile
import zipfile
from pathlib import Path

EHRSHOT_DATA_ROOT = Path.home() / "EHRSHOT"

EXPECTED_FILES = [
    "EHRSHOT_ASSETS.zip",
    "EHRSHOT_MEDS.zip",
    "meds_omop_ehrshot.tar.gz",
    "meds_reader_omop_ehrshot.tar.gz",
]


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _prompt_upload(missing: list[str]) -> None:
    if not _in_colab():
        raise RuntimeError(
            f"Missing {missing} under {EHRSHOT_DATA_ROOT}, and this isn't a Colab "
            "session so there's no upload widget available. Copy the files there "
            "yourself and re-run."
        )
    from google.colab import files

    print(f"Missing from {EHRSHOT_DATA_ROOT}: {missing}")
    print("Select them now (you can select multiple files at once in the dialog).")
    uploaded = files.upload()
    for name, content in uploaded.items():
        (EHRSHOT_DATA_ROOT / name).write_bytes(content)


def _extract_if_needed() -> None:
    marker = EHRSHOT_DATA_ROOT / ".extracted"
    if marker.exists():
        return
    print("Extracting archives (first time this session — takes a few minutes)...")
    for name in EXPECTED_FILES:
        path = EHRSHOT_DATA_ROOT / name
        if not path.exists():
            continue
        if name.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                zf.extractall(EHRSHOT_DATA_ROOT)
        elif name.endswith(".tar.gz"):
            with tarfile.open(path) as tf:
                tf.extractall(EHRSHOT_DATA_ROOT)
    marker.touch()


def _verify_checksums(checksums_path: Path) -> None:
    if not checksums_path.exists():
        return
    expected: dict[str, str] = {}
    for line in checksums_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        expected[name.strip()] = digest.strip()
    if not expected:
        return  # checksums.txt exists but hasn't been filled in yet
    print("Verifying checksums against data/checksums.txt:")
    for name, digest in expected.items():
        path = EHRSHOT_DATA_ROOT / name
        if not path.exists():
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        status = "OK" if actual == digest else "MISMATCH — you may have a different dataset version"
        print(f"  {name}: {status}")


def ensure_ehrshot_data(checksums_path: str | Path | None = None) -> Path:
    """Make sure ~/EHRSHOT has the dataset, prompting for upload if missing,
    extract it if needed, and set EHRSHOT_DATA_ROOT for the rest of the session.
    Returns the resolved path.
    """
    EHRSHOT_DATA_ROOT.mkdir(exist_ok=True)
    missing = [f for f in EXPECTED_FILES if not (EHRSHOT_DATA_ROOT / f).exists()]
    if missing:
        _prompt_upload(missing)
    _extract_if_needed()
    os.environ["EHRSHOT_DATA_ROOT"] = str(EHRSHOT_DATA_ROOT)
    print(f"EHRSHOT_DATA_ROOT = {EHRSHOT_DATA_ROOT}")
    if checksums_path:
        _verify_checksums(Path(checksums_path))
    return EHRSHOT_DATA_ROOT
