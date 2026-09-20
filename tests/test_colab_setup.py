from scripts import colab_setup


def test_reassemble_chunks_concatenates_in_order_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_setup, "EHRSHOT_DATA_ROOT", tmp_path)

    (tmp_path / "EHRSHOT_ASSETS.zip.part_aa").write_bytes(b"AAAA")
    (tmp_path / "EHRSHOT_ASSETS.zip.part_ab").write_bytes(b"BBBB")
    (tmp_path / "EHRSHOT_ASSETS.zip.part_ac").write_bytes(b"CCCC")
    (tmp_path / "EHRSHOT_MEDS.zip").write_bytes(b"UNRELATED")  # not a chunk, must be left alone

    colab_setup._reassemble_chunks()

    reassembled = tmp_path / "EHRSHOT_ASSETS.zip"
    assert reassembled.read_bytes() == b"AAAABBBBCCCC"
    assert not (tmp_path / "EHRSHOT_ASSETS.zip.part_aa").exists()
    assert not (tmp_path / "EHRSHOT_ASSETS.zip.part_ab").exists()
    assert not (tmp_path / "EHRSHOT_ASSETS.zip.part_ac").exists()
    assert (tmp_path / "EHRSHOT_MEDS.zip").read_bytes() == b"UNRELATED"


def test_reassemble_chunks_is_a_no_op_when_nothing_to_reassemble(tmp_path, monkeypatch):
    monkeypatch.setattr(colab_setup, "EHRSHOT_DATA_ROOT", tmp_path)
    (tmp_path / "EHRSHOT_MEDS.zip").write_bytes(b"WHOLE_FILE")

    colab_setup._reassemble_chunks()  # should not raise, should not touch anything

    assert (tmp_path / "EHRSHOT_MEDS.zip").read_bytes() == b"WHOLE_FILE"
