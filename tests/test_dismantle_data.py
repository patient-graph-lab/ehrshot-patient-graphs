from scripts.colab_setup import _CHUNK_PATTERN
from scripts.dismantle_data import SMALL_FILES, dismantle, gather_small_files


def test_dismantle_splits_into_correctly_sized_chunks(tmp_path):
    source = tmp_path / "EHRSHOT_ASSETS.zip"
    source.write_bytes(b"X" * 1000)
    out_dir = tmp_path / "data-chunks"

    parts = dismantle(source, out_dir, chunk_size_bytes=300)

    assert len(parts) == 4  # 300, 300, 300, 100
    assert [p.stat().st_size for p in parts] == [300, 300, 300, 100]
    assert b"".join(p.read_bytes() for p in parts) == b"X" * 1000


def test_dismantle_output_names_match_reassembly_pattern(tmp_path):
    source = tmp_path / "EHRSHOT_ASSETS.zip"
    source.write_bytes(b"Y" * 10)
    out_dir = tmp_path / "data-chunks"

    parts = dismantle(source, out_dir, chunk_size_bytes=4)

    for p in parts:
        m = _CHUNK_PATTERN.match(p.name)
        assert m is not None
        assert m.group("original") == "EHRSHOT_ASSETS.zip"


def test_gather_small_files_copies_present_files_and_reports_missing(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    out_dir = tmp_path / "data-chunks"
    out_dir.mkdir()

    (source_dir / SMALL_FILES[0]).write_bytes(b"meds")
    (source_dir / SMALL_FILES[1]).write_bytes(b"omop")
    # SMALL_FILES[2] deliberately not created — simulates not having it yet

    missing = gather_small_files(source_dir, out_dir)

    assert (out_dir / SMALL_FILES[0]).read_bytes() == b"meds"
    assert (out_dir / SMALL_FILES[1]).read_bytes() == b"omop"
    assert not (out_dir / SMALL_FILES[2]).exists()
    assert missing == [SMALL_FILES[2]]
    # originals untouched, not moved
    assert (source_dir / SMALL_FILES[0]).exists()
