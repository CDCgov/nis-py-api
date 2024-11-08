from sodapy import Socrata
import polars as pl
from datetime import date
import platformdirs
from pathlib import Path


def default_cache_path() -> Path:
    return Path(platformdirs.user_cache_dir("nisapi", ensure_exists=True))


def get_dataset_cache_path(id: str, cache_path: Path = None) -> Path:
    if cache_path is None:
        cache_path = default_cache_path()

    return Path(cache_path) / f"id={id}" / "part-0.parquet"


def cache_dataset(id: str, app_token=None) -> None:
    path = get_dataset_cache_path(id)

    # ensure that there is a directory to save the data to
    data_dir = path.parent
    if data_dir.exist():
        assert data_dir.is_dir()
    else:
        data_dir.mkdir()

    with Socrata("data.cdc.gov", app_token) as client:
        df = client.get(id)

    pl.DataFrame(df).write_parquet(path)


def get_dataset(id: str, app_token=None, cache_path: Path = None) -> pl.DataFrame:
    if cache_path is None:
        cache_path = default_cache_path()

    # check if this id is in the cache
    cached_ids = (
        pl.scan_parquet(cache_path).select(pl.col("id").unique()).collect()["id"]
    )

    if id not in cached_ids:
        cache_dataset(id, app_token=app_token)


def get_dataset_ids(vaccine: str, start_date: date, end_date: date) -> [str]:
    return "sw5n-wg2p"  # flu 2021/2022 through 2024/2025
    # "udsf-9v7b" # COVID-19 updated vaccine


def get_nis(vaccine: str, start_date: date, end_date: date, app_token=None):
    dataset_ids = get_dataset_ids(
        vaccine=vaccine, start_date=start_date, end_date=end_date
    )

    return pl.concat([get_dataset(id) for id in dataset_ids])
