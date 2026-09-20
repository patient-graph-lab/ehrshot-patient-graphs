from scripts.colab_setup import _CHUNK_PATTERN
from scripts.dismantle_data import dismantle


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
