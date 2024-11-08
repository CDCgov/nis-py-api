from sodapy import Socrata
import polars as pl
import platformdirs
from pathlib import Path
from typing import Callable
import warnings
import yaml


def default_cache_path(ensure_exists=False) -> Path:
    return Path(platformdirs.user_cache_dir("nisapi", ensure_exists=ensure_exists))


def dataset_cache_path(id: str, cache_path: Path = None, ensure_exists=False) -> Path:
    if cache_path is None:
        cache_path = default_cache_path(ensure_exists=ensure_exists)

    return Path(cache_path) / f"id={id}" / "part-0.parquet"


def download_dataset(id: str, app_token=None) -> pl.DataFrame:
    with Socrata("data.cdc.gov", app_token) as client:
        df = client.get(id)

    return df


def cache_dataset(
    id: str,
    get_fun: Callable[[str], pl.DataFrame] = download_dataset,
    cache_path: Path = None,
    overwrite: str = "warn",
    **kwargs,
) -> None:
    if cache_path is None:
        cache_path = dataset_cache_path(id, ensure_exists=True)
    else:
        cache_path = Path(cache_path)

    assert overwrite in ["error", "warn", "skip", "yes"]

    if cache_path.exists():
        msg = f"Cached file {cache_path} exists; use force=True to overwrite"
        if overwrite == "error":
            raise RuntimeError(msg)
        elif overwrite == "warn":
            warnings.warn(msg)
        elif overwrite == "skip":
            # quietly skip this file
            return None

    # ensure that there is a directory to save the data to
    data_dir = cache_path.parent
    if data_dir.exists():
        assert data_dir.is_dir()
    else:
        data_dir.mkdir()

    df = get_fun(id, **kwargs)

    pl.DataFrame(df).write_parquet(cache_path)


def get_datasets() -> [dict]:
    with open("nisapi/datasets.yaml") as f:
        return yaml.safe_load(f)


def get_nis(cache: Path = None):
    if cache is None:
        cache = default_cache_path()

    return pl.scan_parquet(cache)
