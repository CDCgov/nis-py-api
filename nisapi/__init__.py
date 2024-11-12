from sodapy import Socrata
import polars as pl
import platformdirs
from pathlib import Path
from typing import Callable
import warnings
import yaml

"""Data schema to be used for all datasets"""
data_schema = pl.Schema(
    [
        ("vaccine", pl.String),
        ("geographic_level", pl.String),
        ("geographic_name", pl.String),
        ("demographic_level", pl.String),
        ("demographic_name", pl.String),
        ("indicator_level", pl.String),
        ("indicator_name", pl.String),
        ("week_ending", pl.Date),
        ("estimate", pl.Float64),
        ("ci_half_width_95pct", pl.Float64),
    ]
)


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


def clean_dataset(id: str, df: pl.DataFrame) -> pl.DataFrame:
    """Clean a raw dataset, applying dataset-specific cleaning rules

    Args:
        id (str): dataset ID
        df (pl.DataFrame): raw dataset

    Returns:
        pl.DataFrame: clean dataset
    """
    if id == "sw5n-wg2p":
        clean = (
            df.rename(
                {
                    "indicator_label": "indicator_level",
                    "indicator_category_label": "indicator_name",
                    "estimates": "estimate",
                }
            )
            .select(data_schema.names())
            .with_columns(
                pl.col(
                    [
                        "vaccine",
                        "geographic_level",
                        "demographic_level",
                        "indicator_name",
                        "indicator_level",
                    ]
                ).str.to_lowercase()
            )
            .with_columns(
                pl.col("geographic_level").replace({"national": "nation"}),
                pl.col("geographic_name").replace({"National": "nation"}),
            )
            # this dataset has an error: for `demographic_level="overall"`, it has
            # `demographic_name="18+ years"`, but it should be "overall"
            .with_columns(
                demographic_name=pl.when(
                    pl.col("demographic_level") == pl.lit("overall")
                )
                .then(pl.lit("overall"))
                .otherwise(pl.col("demographic_name"))
            )
            # this dataset has two indicators: `4-level vaccination and intent` and
            # `Up-to-date` ("Yes" if the 4-level factor is "Received a vaccination";
            # otherwise "No"). The second is redundant, so drop it.
            .filter(pl.col("indicator_level") == "4-level vaccination and intent")
            # cast types
            .with_columns(
                pl.col("week_ending").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f"),
                pl.col(["estimate", "ci_half_width_95pct"]).cast(pl.Float64),
            )
            # change percents to proportions
            .with_columns(pl.col(["estimate", "ci_half_width_95pct"]) / 100.0)
        )

        # check that the date doesn't have any trailing seconds
        assert (clean["week_ending"].dt.truncate("1d") == clean["week_ending"]).all()

        clean = clean.with_columns(pl.col("week_ending").dt.date())
    else:
        raise RuntimeError(f"No cleaning set up for dataset {id}")

    validate(clean)
    return clean


def validate(df: pl.DataFrame):
    """Validate a clean dataset

    Args:
        df (pl.DataFrame): dataset
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # df must have expected column order and types
    assert df.schema == data_schema

    # `vaccine` must be in a certain set
    assert df["vaccine"].is_in(["flu", "covid"]).all()

    # `geographic_level` must be in a certain set
    assert df["geographic_level"].is_in(["nation", "region", "state", "substate"]).all()
    # if `geographic_level` is "nation", `geographic_name` must also be "nation"
    assert (
        df.filter(pl.col("geographic_level") == pl.lit("nation"))["geographic_name"]
        == "nation"
    ).all()
    # if `demographic_level` is "overall", `demographic_name` must also be "overall"
    assert (
        df.filter(pl.col("demographic_level") == pl.lit("overall"))["demographic_name"]
        == "overall"
    ).all()

    # estimates must be percents
    assert df["estimate"].is_between(0.0, 1.0).all()
    # confidence intervals must be non-negative
    assert (df["ci_half_width_95pct"] >= 0.0).all()


def _get_dataset(id: str) -> pl.DataFrame:
    """Download and clean a dataset

    Args:
        id (str): dataset ID

    Returns:
        pl.DataFrame: clean dataset
    """
    raw_df = download_dataset(id)
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
