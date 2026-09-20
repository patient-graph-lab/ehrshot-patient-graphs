"""Split EHRSHOT_ASSETS.zip into upload-sized chunks for Colab.

Run this locally, wherever the dataset files live (not in Colab):

    python scripts/dismantle_data.py

Writes chunks into ./data-chunks/, e.g. EHRSHOT_ASSETS.zip.part_000,
part_001, ... Only EHRSHOT_ASSETS.zip needs this — the other 3 dataset
files are small enough to upload as-is. Reassembled automatically by
ensure_ehrshot_data() in scripts/colab_setup.py after you upload the
chunks plus the other 3 files in one multi-select.

Pure Python, no dependency on the Unix `split` command — works the same
on Windows, Mac, and Linux.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CHUNK_SIZE_MB = 400  # comfortably under Colab's upload-widget reliability ceiling


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", default="EHRSHOT_ASSETS.zip", help="File to split"
    )
    parser.add_argument(
        "--out-dir", default="data-chunks", help="Where to write the pieces"
    )
    parser.add_argument(
        "--chunk-size-mb", type=int, default=DEFAULT_CHUNK_SIZE_MB, help="Chunk size in MB"
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Not found: {source}")

    parts = dismantle(source, Path(args.out_dir), args.chunk_size_mb * 1024 * 1024)
    print(f"\nDone: {len(parts)} pieces in {args.out_dir}/")
    print(
        "Upload every piece in that folder, plus the other 3 dataset files, "
        "in one multi-select when the Colab notebook prompts for upload."
    )


if __name__ == "__main__":
    main()
