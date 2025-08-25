import importlib.resources
import shutil
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional

import platformdirs
import polars as pl
import json

import nisapi.clean
import nisapi.socrata


def get_nis(path: Optional[Path] = None) -> pl.LazyFrame:
    """Get the cleaned NIS dataset

    Args:
        path (Path, optional): Path to cache. If None (default), use
            default location.

    Returns:
        pl.LazyFrame: _description_
    """
    # read the clean data
    return pl.scan_parquet(get_data_path(path=path))


def cache_all_datasets(
    path: Optional[Path] = None,
    app_token: Optional[str] = None,
    overwrite: str = "warn",
    validation_mode: str = "warn",
) -> None:
    """Download all raw datasets known in the metadata, and clean them

    Args:
        path (Path, optional): Path to cache. If None (default), use
            default location.
        app_token (str): Socrata developer API token
        overwrite (str): Overwrite existing datasets? Default ("warn")
            will not overwrite but will print a warning.
        validation_mode (str): How should validation problems be handled?
            Default ("warn") will print a warning and continue.
    """
    if path is None:
        path = default_cache_path()

    for id in _get_dataset_ids():
        _cache_clean_dataset(
            id,
            cache_path=path,
            app_token=app_token,
            overwrite=overwrite,
            validation_mode=validation_mode,
        )


def delete_cache(path: Optional[Path] = None, confirm: bool = True) -> None:
    """Delete cache

    Args:
        path (Path, optional): Path to cache. If None (default), use
            default location.
        confirm (bool, optional): If True (the default), get interactive
          confirmation before deleting
    """
    if path is None:
        path = default_cache_path()

    if not Path(path).exists():
        warnings.warn(f"Cache path {path} does not exist")
        return None

    if confirm:
        yn = input(f"Remove all files in {path}? [y/N] ").lower()

    if confirm and yn != "y":
        print("Not deleting cache")
    elif not confirm or yn == "y":
        shutil.rmtree(path)


def _get_dataset_ids() -> Sequence[str]:
    with importlib.resources.open_text(nisapi, "datasets.json") as f:
        datasets = json.load(f)

    return [dataset["id"] for dataset in datasets["datasets"]]


def _cache_clean_dataset(
    id: str,
    cache_path: Path,
    app_token: Optional[str],
    overwrite: str,
    validation_mode: str,
) -> None:
    with importlib.resources.open_text(nisapi, "datasets.json") as f:
        datasets = json.load(f)

    clean_args = [d for d in datasets["datasets"] if d.get("id") == id][0].get(
        "cleaning_arguments"
    )

    raw_data_path = get_data_path(path=cache_path, type_="raw", id=id)
    raw_data = _get_nis_raw(id=id, raw_data_path=raw_data_path, app_token=app_token)
    clean_data = nisapi.clean.clean_dataset(
        df=raw_data, id=id, clean_args=clean_args, validation_mode=validation_mode
    )
    clean_data_path = get_data_path(path=cache_path, type_="clean", id=id)
    clean_data_filepath = clean_data_path / "part-0.parquet"

    if clean_data_filepath.exists():
        msg = f"Clean dataset {clean_data_filepath} already exists"
        if overwrite == "warn":
            warnings.warn(msg)
            return None
        else:
            raise RuntimeError(f"Invalid overwrite option '{overwrite}'")

    if not clean_data_path.exists():
        clean_data_path.mkdir(parents=True)

    clean_data.write_parquet(clean_data_filepath)


def get_data_path(
    path: Optional[Path] = None,
    type_: Literal["clean", "raw"] = "clean",
    id: Optional[str] = None,
) -> Path:
    """
    Get the path to a particular dataset, or to a part of the cache.

    Args:
        path (Path, optional): Path to the root cache directory. If None,
            use the default location from default_cache_path().
        type_ (str): Type of dataset. One of "clean" or "raw". Default
            is "clean".
        id (str, optional): Dataset ID. If None, return a path to the top level of
            part of the cache. If provided, return the path to
            that particular dataset in the cache.

    Returns:
        Path: path to data directory
    """
    if path is None:
        path = default_cache_path()

    assert type_ in ["clean", "raw"], f"Unknown cache type: {type_}"

    if id is None:
        return path / type_
    else:
        return path / type_ / f"id={id}"


def default_cache_path() -> Path:
    """
    Get the default path to the cache directory.

    Returns:
        Path: Default path to the cache directory
    """
    return Path(platformdirs.user_cache_dir("nisapi"))


def _get_nis_raw(
    id: str, raw_data_path: Path, app_token: Optional[str]
) -> pl.LazyFrame:
    """
    Args:
        id (str): dataset ID
        raw_data_path (Path): path to the raw dataset directory
        app_token (str, optional): Socrata developer API token, or None
    """
    filepath = raw_data_path / "part-0.parquet"

    if not raw_data_path.exists():
        raw_data_path.mkdir(parents=True)

    if not filepath.exists():
        data = _download_dataset(id=id, app_token=app_token)
        data.write_parquet(filepath)

    return pl.scan_parquet(filepath)


def _download_dataset(id: str, app_token: Optional[str]) -> pl.DataFrame:
    """Download a raw NIS dataset

    Args:
        id (str): dataset ID
        app_token (str, optional): Socrata developer API token, or None

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
