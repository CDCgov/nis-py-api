import nisapi
from pathlib import Path
import polars as pl
import tempfile


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
        nisapi.cache_dataset(id="BOGUS", get_fun=get_fun, cache_path=f.name)

        assert Path(f.name).exists()
