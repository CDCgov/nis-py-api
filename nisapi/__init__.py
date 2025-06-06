import importlib.resources
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Sequence

import platformdirs
import polars as pl
import yaml

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
    if path is None:
        path = Path(root_cache_path(), "clean")

    return pl.scan_parquet(path)


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
        path = root_cache_path()

    for id in _get_dataset_ids():
        _cache_clean_dataset(
            id,
            root_path=path,
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
        path = root_cache_path()

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
    with importlib.resources.open_text(nisapi, "datasets.yaml") as f:
        metadata = yaml.safe_load(f)

    return [dataset["id"] for dataset in metadata]


def _cache_clean_dataset(
    id: str,
    root_path: Path,
    app_token: Optional[str],
    overwrite: str,
    validation_mode: str,
) -> None:
    raw_data = _get_nis_raw(id, root_path=root_path, app_token=app_token)
    clean_data = nisapi.clean.clean_dataset(
        df=raw_data, id=id, validation_mode=validation_mode
    )
    clean_path_dir = _dataset_cache_path(root_path=root_path, type_="clean", id=id)
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


def root_cache_path() -> Path:
    return Path(platformdirs.user_cache_dir("nisapi"))


def _dataset_cache_path(root_path: Path, type_: str, id: str) -> Path:
    """Construct path to a particular dataset in the cache

    Cache starts at the "root", goes through either "raw" or "clean",
    and then has a subdirectory for each dataset ID.

    Args:
        root_path (Path): Top-level cache directory
        type_ (str): Either "raw" or "clean"
        id (str): Dataset ID

    Returns:
        Path: path to the dataset
    """
    return Path(root_path, type_, f"id={id}")


def _get_nis_raw(id: str, root_path: Path, app_token: Optional[str]) -> pl.LazyFrame:
    dir_path = _dataset_cache_path(root_path=root_path, type_="raw", id=id)
    path = dir_path / "part-0.parquet"

    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    if not path.exists():
        data = _download_dataset(id=id, app_token=app_token)
        data.write_parquet(path)

    return pl.scan_parquet(dir_path)


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
