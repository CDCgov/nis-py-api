import polars as pl
import platformdirs
from pathlib import Path
import yaml
import shutil
import nisapi.clean
import nisapi.socrata
from typing import Sequence
import warnings
import tempfile


def get_nis(path: Path = None) -> pl.LazyFrame:
    if path is None:
        cache = Path(_root_cache_path(), "clean")

    return pl.scan_parquet(cache)


def cache_all_datasets(app_token: str = None) -> None:
    """Download all raw datasets known in the metadata, and clean them

    Args:
        app_token (str)
    """
    for id in _get_dataset_ids():
        _cache_clean_dataset(id, app_token=app_token)


def delete_cache(cache_path: str = None, confirm: bool = True) -> None:
    """Delete cache

    Args:
        cache_path (str, optional): Path to cache. If None (default), get
          path from `_root_cache_path()`.
        confirm (bool, optional): If True (the default), get interactive
          confirmation before deleting
    """
    if cache_path is None:
        cache_path = _root_cache_path()

    if not Path(cache_path).exists():
        warnings.warn(f"Cache path {cache_path} does not exist")
        return None

    if confirm:
        yn = input(f"Remove all files in {cache_path}? [y/N] ").lower()

    if confirm and yn != "y":
        print("Not deleting cache")
    elif not confirm or yn == "y":
        shutil.rmtree(cache_path)


def _get_dataset_ids() -> Sequence[str]:
    with open("nisapi/datasets.yaml") as f:
        metadata = yaml.safe_load(f)

    return [dataset["id"] for dataset in metadata]


def _cache_clean_dataset(
    id: str, app_token: str = None, overwrite: str = "warn"
) -> None:
    raw_data = _get_nis_raw(id, app_token=app_token)
    clean_data = nisapi.clean.clean_dataset(df=raw_data, id=id)
    clean_path_dir = _dataset_cache_path(
        root_path=_root_cache_path(), type_="clean", id=id
    )
    clean_path = clean_path_dir / "part-0.parquet"

    if clean_path.exists():
        msg = f"Clean dataset {clean_path} already exists"
        if overwrite == "warn":
            warnings.warn(msg)
            return None
        else:
            raise RuntimeError(f"Invalid overwrite option '{overwrite}'")

    if not clean_path_dir.exists():
        clean_path_dir.mkdir(parents=True)

    clean_data.write_parquet(clean_path)


def _root_cache_path() -> Path:
    return Path(platformdirs.user_cache_dir("nisapi"))


def _dataset_cache_path(root_path: str, type_: str, id: str) -> Path:
    return Path(root_path, type_, f"id={id}")


def _get_nis_raw(id: str, app_token: str = None) -> pl.LazyFrame:
    root_path = _root_cache_path()
    dir_path = _dataset_cache_path(root_path=root_path, type_="raw", id=id)
    path = dir_path / "part-0.parquet"

    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    if not path.exists():
        data = _download_dataset(id=id, app_token=app_token)
        data.write_parquet(path)

    return pl.scan_parquet(dir_path)


def _download_dataset(id: str, app_token: str = None) -> pl.DataFrame:
    """Download a raw NIS dataset

    Args:
        id (str): dataset ID
        app_token (str, optional): Socrata developer API token

    Returns:
        pl.DataFrame: raw dataset
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pages = nisapi.socrata.download_dataset_pages(id, app_token=app_token)
        for i, page in enumerate(pages):
            path = f"part-{i}.parquet"
            df = pl.DataFrame(page)
            df.write_parquet(Path(tmpdir) / path)

        return pl.read_parquet(tmpdir)
