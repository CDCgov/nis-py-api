from sodapy import Socrata
import polars as pl
import platformdirs
from pathlib import Path
from typing import Callable


def default_cache_path() -> Path:
    return Path(platformdirs.user_cache_dir("nisapi", ensure_exists=True))


def get_dataset_cache_path(id: str, cache_path: Path = None) -> Path:
    if cache_path is None:
        cache_path = default_cache_path()

    return Path(cache_path) / f"id={id}" / "part-0.parquet"


def download_dataset(id: str, app_token=None) -> pl.DataFrame:
    with Socrata("data.cdc.gov", app_token) as client:
        df = client.get(id)

    return df


def cache_dataset(
    id: str, get_fun: Callable[[str], pl.DataFrame] = download_dataset, **kwargs
) -> None:
    path = get_dataset_cache_path(id)

    # ensure that there is a directory to save the data to
    data_dir = path.parent
    if data_dir.exists():
        assert data_dir.is_dir()
    else:
        data_dir.mkdir()

    df = get_fun(id, **kwargs)

    pl.DataFrame(df).write_parquet(path)


def get_nis(cache: Path = None):
    if cache is None:
        cache = default_cache_path()

    return pl.scan_parquet(cache)
