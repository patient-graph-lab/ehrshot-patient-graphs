import os
import tarfile
import zipfile

import pytest

from scripts.local_setup import ensure_local_data


def test_ensure_local_data_extracts_only_present_archives(tmp_path, monkeypatch):
    monkeypatch.delenv("EHRSHOT_DATA_ROOT", raising=False)
    zip_path = tmp_path / "EHRSHOT_ASSETS.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("EHRSHOT_ASSETS/benchmark/dummy.txt", "hello")

    result = ensure_local_data(tmp_path)

    assert result == tmp_path
    assert (tmp_path / "EHRSHOT_ASSETS" / "benchmark" / "dummy.txt").read_text() == "hello"
    # None of the other 3 archives were present -- must not error, must not
    # create their directories
    assert not (tmp_path / "meds_reader_omop_ehrshot").exists()
    assert os.environ["EHRSHOT_DATA_ROOT"] == str(tmp_path)


def test_ensure_local_data_extracts_tar_archive_too(tmp_path, monkeypatch):
    monkeypatch.delenv("EHRSHOT_DATA_ROOT", raising=False)
    inner = tmp_path / "meds_reader_omop_ehrshot"
    inner.mkdir()
    (inner / "subject_id").write_text("fake")
    tar_path = tmp_path / "meds_reader_omop_ehrshot.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(inner, arcname="meds_reader_omop_ehrshot")
    # Remove the source dir so we can tell extraction actually happened
    (inner / "subject_id").unlink()
    inner.rmdir()

    ensure_local_data(tmp_path)

    assert (tmp_path / "meds_reader_omop_ehrshot" / "subject_id").exists()


def test_ensure_local_data_skips_extraction_if_target_dir_already_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("EHRSHOT_DATA_ROOT", raising=False)
    zip_path = tmp_path / "EHRSHOT_ASSETS.zip"
    with zipfile.ZipFile(zip_path, "w"):
        pass  # empty, would extract to nothing
    target = tmp_path / "EHRSHOT_ASSETS"
    target.mkdir()
    (target / "already-here.txt").write_text("untouched")

    ensure_local_data(tmp_path)

    # If it had re-extracted the empty zip, nothing would change either way,
    # but this confirms no error occurs when the target dir pre-exists
    assert (target / "already-here.txt").read_text() == "untouched"


def test_ensure_local_data_raises_if_nothing_present(tmp_path, monkeypatch):
    monkeypatch.delenv("EHRSHOT_DATA_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="EHRSHOT_ASSETS.zip"):
        ensure_local_data(tmp_path)
