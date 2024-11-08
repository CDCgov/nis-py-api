import nisapi
from pathlib import Path
import polars as pl
import tempfile
import pytest


def test_default_cache_path():
    path = nisapi.default_cache_path(ensure_exists=False)
    assert isinstance(path, Path)


def test_dataset_cache_path():
    path = nisapi.dataset_cache_path(
        id="1234", cache_path="fake_path", ensure_exists=False
    )
    assert path == Path("fake_path", "id=1234", "part-0.parquet")


def test_cache_dataset():
    def get_fun(id):
        return pl.read_parquet(Path("tests", "data", "test.parquet"))

    with tempfile.NamedTemporaryFile() as f:
        nisapi.cache_dataset(
            id="BOGUS", get_fun=get_fun, cache_path=f.name, overwrite="yes"
        )

        assert Path(f.name).exists()


def test_cache_dataset_overwrite_error():
    """If force=False and strict="""

    def get_fun(id):
        return pl.read_parquet(Path("tests", "data", "test.parquet"))

    with tempfile.NamedTemporaryFile() as f:
        nisapi.cache_dataset(
            id="BOGUS", get_fun=get_fun, cache_path=f.name, overwrite="yes"
        )

        with pytest.raises(RuntimeError, match="exists"):
            nisapi.cache_dataset(
                id="BOGUS", get_fun=get_fun, cache_path=f.name, overwrite="error"
            )


def test_cache_dataset_overwrite_warning():
    """If force=False and strict="""

    def get_fun(id):
        return pl.read_parquet(Path("tests", "data", "test.parquet"))

    with tempfile.NamedTemporaryFile() as f:
        nisapi.cache_dataset(
            id="BOGUS", get_fun=get_fun, cache_path=f.name, overwrite="yes"
        )

        with pytest.warns(UserWarning, match="exists"):
            nisapi.cache_dataset(
                id="BOGUS", get_fun=get_fun, cache_path=f.name, overwrite="warn"
            )


def test_cache_dataset_overwrite_skip():
    """If force=False and strict=False, cache_dataset should quietly not overwrite anything"""

    def get_fun(id):
        return pl.read_parquet(Path("tests", "data", "test.parquet"))

    with tempfile.NamedTemporaryFile() as f:
        f.write(b"fake data")

        nisapi.cache_dataset(
            id="bogus_id", get_fun=get_fun, cache_path=f.name, overwrite="skip"
        )

        f.seek(0)
        assert f.read() == b"fake data"
