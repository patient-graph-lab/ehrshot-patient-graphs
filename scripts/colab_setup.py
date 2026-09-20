"""Bootstrap EHRSHOT data access for a Colab session.

Convention for this project: data always lives at ~/EHRSHOT, re-populated fresh
every session, and never committed to git. Nothing about the dataset persists
on the VM after the session ends — see DATA_ACCESS.md for why that matters.

Two ways to get the dataset onto the VM's local disk:

1. **Recommended: `source_dir=` pointing at a mounted Google Drive folder,
   used as a brief pass-through.** Drive's own uploader is a real resumable
   HTTP upload; Colab's direct-upload paths are not (see #2) — measured Sep
   20 on a 366 Mbps connection: ~90 seconds via Drive vs. an estimated 2.5+
   hours via direct upload for the full ~4.95GB dataset. Upload the 4 files
   to a Drive folder once, then this function copies them to the VM's local
   disk; call `delete_from_drive(source_dir)` immediately after — that's
   what keeps it a brief pass-through rather than a standing copy. This
   does mean the data touches Drive for a few minutes; see DATA_ACCESS.md
   for the DUA reasoning behind why that's been judged acceptable.
2. **Fallback (no `source_dir` given): chunked browser upload, zero
   third-party storage touched.** Colab's upload paths (the
   `files.upload()` widget, and the file-browser drag-and-drop pane) relay
   through a base64-encoded message channel to the kernel — measured
   throughput is roughly 4-5 Mbps *regardless of actual bandwidth or VM
   RAM*, so a multi-GB dataset takes hours even on a fast connection. Of
   the 4 dataset files, only `EHRSHOT_ASSETS.zip` (4.3GB) needs splitting
   first — `python dismantle_data.py EHRSHOT_ASSETS.zip` writes
   `./data-chunks/` with the pieces plus copies of the other 3 files, then
   upload that whole folder's contents in one `files.upload()` call; this
   function detects and reassembles any `*.part_*` files automatically.
   Use this only if avoiding Drive entirely matters more than speed.

Usage, as the first real cell of any notebook (after cloning the repo and
running `pip install -e ".[dev,data]"`):

    from scripts.colab_setup import ensure_ehrshot_data
    data_root = ensure_ehrshot_data(checksums_path="data/checksums.txt")
"""

from __future__ import annotations

import hashlib
import os
import re
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

_CHUNK_PATTERN = re.compile(r"^(?P<original>.+)\.part_[a-zA-Z0-9]+$")


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
    if "EHRSHOT_ASSETS.zip" in missing:
        print(
            "EHRSHOT_ASSETS.zip is 4.3GB — too large to upload as one file reliably.\n"
            "On your own machine, not in Colab:\n"
            "  python dismantle_data.py EHRSHOT_ASSETS.zip\n"
            "This writes ./data-chunks/ with everything you need. In the dialog "
            "below, open that folder and select everything in it — nothing "
            "outside it. The parts get reassembled automatically."
        )
    print("Select the files now (you can select multiple at once in the dialog).")
    uploaded = files.upload()
    for name, content in uploaded.items():
        (EHRSHOT_DATA_ROOT / name).write_bytes(content)
    _reassemble_chunks()


def _reassemble_chunks() -> None:
    """Find any <name>.part_* files (from `split`) and concatenate them back
    into <name>, in sorted order, then remove the parts. `split`'s default
    suffixes (aa, ab, ... or 000, 001, ...) both sort correctly as strings."""
    parts_by_original: dict[str, list[Path]] = {}
    for f in sorted(EHRSHOT_DATA_ROOT.iterdir()):
        m = _CHUNK_PATTERN.match(f.name)
        if m:
            parts_by_original.setdefault(m.group("original"), []).append(f)
    for original_name, parts in parts_by_original.items():
        parts.sort()
        target = EHRSHOT_DATA_ROOT / original_name
        print(f"Reassembling {len(parts)} parts into {original_name}...")
        with target.open("wb") as out:
            for p in parts:
                out.write(p.read_bytes())
                p.unlink()


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

    Recommended: pass `source_dir` (e.g. a mounted Drive folder) — copies
    missing files from there, ~100x faster than the alternative (see module
    docstring). Without it, falls back to a chunked browser upload prompt
    (zero third-party storage touched, but slow regardless of bandwidth).
    """
    EHRSHOT_DATA_ROOT.mkdir(exist_ok=True)
    missing = [f for f in EXPECTED_FILES if not (EHRSHOT_DATA_ROOT / f).exists()]
    if missing:
        if source_dir:
            _copy_from_source(missing, Path(source_dir))
        else:
            _prompt_upload(missing)
        still_missing = [f for f in EXPECTED_FILES if not (EHRSHOT_DATA_ROOT / f).exists()]
        if still_missing:
            raise FileNotFoundError(
                f"Still missing after upload: {still_missing}. If this includes "
                "EHRSHOT_ASSETS.zip, check every .part_* piece was selected in "
                "the upload dialog — reassembly needs all of them present."
            )
    _extract_if_needed()
    os.environ["EHRSHOT_DATA_ROOT"] = str(EHRSHOT_DATA_ROOT)
    print(f"EHRSHOT_DATA_ROOT = {EHRSHOT_DATA_ROOT}")
    if checksums_path:
        _verify_checksums(Path(checksums_path))
    return EHRSHOT_DATA_ROOT
