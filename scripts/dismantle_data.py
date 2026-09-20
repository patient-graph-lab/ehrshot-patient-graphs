"""Prepare the EHRSHOT dataset for direct Colab upload: split the one file
that's too large, and gather everything that needs uploading into one folder.

This is the only supported way to get the dataset into Colab right now —
Google Drive is confirmed disallowed by Stanford's data stewards as of
Sep 20, 2026 (see DATA_ACCESS.md), so there is no faster alternative to
fall back to at the moment. Colab's direct-upload paths cap out around
4-5 Mbps regardless of your actual bandwidth (measured Sep 20 on a 366
Mbps connection), so the full dataset takes hours even split into pieces.
Stanford's actual recommendation is to develop inside Redivis instead of
Colab, which would remove this problem entirely — see DATA_ACCESS.md for
status before assuming this script is the long-term answer.

Run this locally, wherever the dataset files live (not in Colab), from the
same directory as the 4 dataset files:

    python scripts/dismantle_data.py

Writes ./data-chunks/, containing:
  - EHRSHOT_ASSETS.zip.part_000, part_001, ... (the 4.3GB file, split into
    pieces — the only one of the 4 files too large for Colab's upload
    widget to handle reliably)
  - copies of the other 3 dataset files, unchanged (small enough to upload
    as-is, copied here just so everything needed lives in one folder)

The original EHRSHOT_ASSETS.zip is left untouched in place, deliberately
outside data-chunks/ — so when you're selecting files to upload in Colab,
there's exactly one folder to open, and the 4.3GB original (which would
just hit the same upload problem again) is never sitting there to
accidentally re-select.

Reassembled automatically by ensure_ehrshot_data() in scripts/colab_setup.py
once you upload everything in data-chunks/.

Pure Python, no dependency on the Unix `split` command — works the same on
Windows, Mac, and Linux.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_CHUNK_SIZE_MB = 400  # comfortably under Colab's upload-widget reliability ceiling

LARGE_FILE = "EHRSHOT_ASSETS.zip"
SMALL_FILES = [
    "EHRSHOT_MEDS.zip",
    "meds_omop_ehrshot.tar.gz",
    "meds_reader_omop_ehrshot.tar.gz",
]


def dismantle(source: Path, out_dir: Path, chunk_size_bytes: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    with source.open("rb") as f:
        i = 0
        while True:
            chunk = f.read(chunk_size_bytes)
            if not chunk:
                break
            part_path = out_dir / f"{source.name}.part_{i:03d}"
            part_path.write_bytes(chunk)
            parts.append(part_path)
            print(f"Wrote {part_path} ({len(chunk) / 1e6:.1f} MB)")
            i += 1
    return parts


def gather_small_files(source_dir: Path, out_dir: Path) -> list[str]:
    """Copy the 3 small dataset files into out_dir alongside the split
    pieces, so everything needed for upload lives in one folder. Returns
    the names of any that weren't found (not fatal — you might run this
    before downloading everything)."""
    missing = []
    for name in SMALL_FILES:
        src = source_dir / name
        if not src.exists():
            missing.append(name)
            continue
        dest = out_dir / name
        shutil.copy2(src, dest)
        print(f"Copied {name} ({src.stat().st_size / 1e6:.1f} MB)")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", default=LARGE_FILE, help="Path to EHRSHOT_ASSETS.zip"
    )
    parser.add_argument(
        "--out-dir", default="data-chunks", help="Where to gather everything for upload"
    )
    parser.add_argument(
        "--chunk-size-mb", type=int, default=DEFAULT_CHUNK_SIZE_MB, help="Chunk size in MB"
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Not found: {source}")
    out_dir = Path(args.out_dir)

    parts = dismantle(source, out_dir, args.chunk_size_mb * 1024 * 1024)
    missing = gather_small_files(source.parent, out_dir)

    print(f"\nDone: {out_dir}/ contains {len(parts)} pieces of {source.name} "
          f"+ {len(SMALL_FILES) - len(missing)}/{len(SMALL_FILES)} of the small files.")
    if missing:
        print(f"Not found next to {source}, add these to {out_dir}/ yourself: {missing}")
    print(f"In Colab: open {out_dir}/ and select everything in it — nothing outside it.")


if __name__ == "__main__":
    main()
