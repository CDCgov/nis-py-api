from sodapy import Socrata
import polars as pl
import platformdirs
from pathlib import Path
from typing import Callable
import warnings
import yaml
from .clean import clean_dataset


def default_cache_path(ensure_exists=False) -> Path:
    return Path(platformdirs.user_cache_dir("nisapi", ensure_exists=ensure_exists))


def dataset_cache_path(id: str, cache_path: Path = None, ensure_exists=False) -> Path:
    if cache_path is None:
        cache_path = default_cache_path(ensure_exists=ensure_exists)

    return Path(cache_path) / f"id={id}" / "part-0.parquet"


def download_dataset(id: str, app_token=None) -> pl.DataFrame:
    with Socrata("data.cdc.gov", app_token) as client:
        rows = list(client.get_all(id))

    return pl.DataFrame(rows)


def _get_dataset(id: str, app_token=None) -> pl.DataFrame:
    """Download and clean a dataset

    Args:
        id (str): dataset ID

    Returns:
        pl.DataFrame: clean dataset
    """
    raw_df = download_dataset(id, app_token=app_token)
    clean_df = clean_dataset(id, raw_df)
    return clean_df


def cache_dataset(
    id: str,
    overwrite: str = "warn",
    cache_path: Path = None,
    get_fun: Callable[[str], pl.DataFrame] = _get_dataset,
    **kwargs,
) -> None:
    """Download, clean, and cache a dataset

    Args:
        id (str): dataset ID
        overwrite (str, optional): If "warn" (default), will warn if the cache file already exists. If
            "error", will raise an error. If "skip", will silently do nothing. If "yes", will silently
            overwrite.
        cache_path (Path, optional): Path to cache this specific dataset. If `None` (the default), uses
            the default path.
        get_fun (Callable[[str], pl.DataFrame], optional): Function used to get the raw data. Defaults
            to `_get_dataset`. No reason to change except for testing purposes.

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
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
    df.write_parquet(cache_path)


def get_datasets() -> [dict]:
    with open("nisapi/datasets.yaml") as f:
        return yaml.safe_load(f)


def get_nis(cache: Path = None) -> pl.LazyFrame:
    if cache is None:
        cache = default_cache_path()

    return pl.scan_parquet(cache)
