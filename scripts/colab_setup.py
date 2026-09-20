"""Bootstrap EHRSHOT data access for a Colab session.

Convention for this project: data always lives at ~/EHRSHOT, re-populated fresh
every session, and never committed to git. Nothing about the dataset persists
on the VM after the session ends — see DATA_ACCESS.md for why that matters.

Two ways to get the dataset onto the VM's local disk:

1. Browser upload widget (default, no `source_dir` given). Fine for small
   files, but google.colab.files.upload() moves data through a
   base64-encoded message channel, not a real streaming upload — it can sit
   at 0% indefinitely on a multi-GB file like EHRSHOT_ASSETS.zip. Don't rely
   on it for the full dataset.
2. `source_dir=` pointing at a mounted Google Drive folder used as a brief
   pass-through: Drive's own uploader is a real resumable upload, so getting
   the files there is fast and reliable; this function then copies them from
   Drive to the VM's local disk (fast — same Google network). Call
   `delete_from_drive(source_dir)` immediately after to remove them from
   Drive so nothing persists there. See DATA_ACCESS.md before using this —
   the data touches Drive for a few minutes even though nothing persists,
   which is a different exposure than pure re-upload and worth checking
   against your DUA terms first.

Usage, as the first real cell of any notebook (after cloning the repo and
running `pip install -e ".[dev,data]"`):

    from scripts.colab_setup import ensure_ehrshot_data, delete_from_drive
    data_root = ensure_ehrshot_data(
        checksums_path="data/checksums.txt",
        source_dir="/content/drive/MyDrive/ehrshot_transfer",
    )
    delete_from_drive("/content/drive/MyDrive/ehrshot_transfer")
"""

from __future__ import annotations

import hashlib
import os
import shutil
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


def _copy_from_source(missing: list[str], source_dir: Path) -> None:
    still_missing = []
    for name in missing:
        src = source_dir / name
        if not src.exists():
            still_missing.append(name)
            continue
        size_gb = src.stat().st_size / 1e9
        print(f"Copying {name} ({size_gb:.2f} GB) from {source_dir}...")
        shutil.copy2(src, EHRSHOT_DATA_ROOT / name)
    if still_missing:
        raise FileNotFoundError(
            f"Not found in {source_dir}: {still_missing}\n\n"
            "This is a manual step that happens outside this notebook, in a browser:\n"
            "  1. Go to drive.google.com\n"
            f"  2. In 'My Drive', create a folder named exactly '{source_dir.name}'\n"
            "  3. Upload the missing file(s) into it\n"
            "  4. Wait until Drive's UI shows the upload fully complete, not just started\n"
            "  5. Re-run this cell\n"
            "A Colab cell has no access to your device's filesystem — this upload can't "
            "be done from code, only from Drive's own web/app interface."
        )


def delete_from_drive(source_dir: str | Path) -> None:
    """Remove the dataset files from the Drive pass-through directory once
    ensure_ehrshot_data(source_dir=...) has confirmed they're copied to the
    VM's local disk. Call this right after — it's what makes the Drive
    exposure brief rather than a standing copy."""
    source_dir = Path(source_dir)
    for name in EXPECTED_FILES:
        f = source_dir / name
        if f.exists():
            f.unlink()
            print(f"Deleted {f}")


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


def ensure_ehrshot_data(
    checksums_path: str | Path | None = None,
    source_dir: str | Path | None = None,
) -> Path:
    """Make sure ~/EHRSHOT has the dataset, extract it if needed, and set
    EHRSHOT_DATA_ROOT for the rest of the session. Returns the resolved path.

    If `source_dir` is given (e.g. a mounted Drive folder), copies missing
    files from there instead of prompting the browser-upload widget — see
    the module docstring for why that matters for a file this large.
    """
    EHRSHOT_DATA_ROOT.mkdir(exist_ok=True)
    missing = [f for f in EXPECTED_FILES if not (EHRSHOT_DATA_ROOT / f).exists()]
    if missing:
        if source_dir:
            _copy_from_source(missing, Path(source_dir))
        else:
            _prompt_upload(missing)
    _extract_if_needed()
    os.environ["EHRSHOT_DATA_ROOT"] = str(EHRSHOT_DATA_ROOT)
    print(f"EHRSHOT_DATA_ROOT = {EHRSHOT_DATA_ROOT}")
    if checksums_path:
        _verify_checksums(Path(checksums_path))
    return EHRSHOT_DATA_ROOT
