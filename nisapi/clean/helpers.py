import uuid
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import polars as pl

"""Data schema to be used for all datasets"""
data_schema = pl.Schema(
    [
        ("vaccine", pl.String),
        ("geography_type", pl.String),
        ("geography", pl.String),
        ("domain_type", pl.String),
        ("domain", pl.String),
        ("indicator_type", pl.String),
        ("indicator", pl.String),
        ("time_type", pl.String),
        ("time_start", pl.Date),
        ("time_end", pl.Date),
        ("estimate", pl.Float64),
        ("lci", pl.Float64),
        ("uci", pl.Float64),
        ("sample_size", pl.UInt32),
    ]
)

"""First-level administrative divisions of the US: states, territories, and DC"""
admin1_values = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
    "District of Columbia",
    "Guam",
    "Puerto Rico",
    "U.S. Virgin Islands",
]


def drop_bad_rows(df: pl.LazyFrame, colname: str) -> pl.LazyFrame:
    """
    Bad rows are those with a suppression flag or null values.
    """
    df = df.filter(pl.col(colname) == pl.lit("0")).drop(colname)
    null_rows = df.filter(pl.any_horizontal(pl.all().is_null())).collect()
    if null_rows.shape[0] > 0:
        warnings.warn("Some rows contain null values. These rows will be dropped.")
        print(null_rows)
    df = df.drop_nulls()

    return df


def clean_geography_type(df: pl.LazyFrame, colname: str) -> pl.LazyFrame:
    """
    Geography type is the scale of geographic division.
    Add to the `replace_strict` dictonary as necessary to standardize verbiage.
    """
    df.rename({colname: "geography_type"})
    df = df.with_columns(
        pl.col("geography_type").str.to_lowercase().str.strip_chars()
    ).with_columns(
        pl.col("geography_type").replace_strict(
            {
                "national": "nation",
                "nation": "nation",
                "state": "admin1",
                "region": "region",
                "hhs region": "region",
                "substate": "substate",
                "local": "local",
            }
        )
    )

    return df


def clean_geography(df: pl.LazyFrame, colname: str) -> pl.LazyFrame:
    """
    Geography is the specific geographic location.
    Add to the `replace` dictionary as necessary to standardize verbiage.
    """
    df.rename({colname: "geography"})
    df = df.with_columns(
        pl.col("geography").str.strip_chars().replace({"National": "nation"})
    )

    return df


def clean_domain_type(df: pl.LazyFrame, colname: str) -> pl.LazyFrame:
    """
    Domain type is the demographic feature used to define groups.
    Add to the `replace` dictionary as necessary to standardize verbiage.
    """
    df.rename({colname: "domain_type"})
    df = df.with_columns(
        pl.col("domain_type").str.to_lowercase().str.strip_chars()
    ).with_columns(pl.col("domain_type").replace({"overall": "age"}))

    return df


def clean_domain(df: pl.LazyFrame, colname: str) -> pl.LazyFrame:
    """
    Domain is the specific demographic group.
    """
    df.rename({colname: "domain"})
    df = df.with_columns(pl.col("domain").str.strip_chars())

    return df


def clean_indicator_type(df: pl.LazyFrame, colname: str) -> pl.LazyFrame:
    """
    Indicator type is the survey question that was asked.
    """
    df.rename({colname: "indicator_type"})
    df = df.with_columns(pl.col("indicator_type").str.to_lowercase().str.strip_chars())

    return df


def clean_indicator(
    df: pl.LazyFrame, colname: str, synonyms: Optional[List[Tuple[str, str]]] = None
) -> pl.LazyFrame:
    """
    Indicator is the specific answer to the survey question.
    Synonyms is a list of (indicator_type, indicator) tuples that are identical.
    The first synonym will be kept and all others dropped.
    E.g. ("4-level vaccination and intent", "received a vaccinated") and ("up-to-date", "yes")
    are synonymous, so the former should be kept and the latter discarded.
    """
    df.rename({colname: "indicator"})
    df = df.with_columns(pl.col("indicator").str.strip_chars())
    if synonyms is not None:
        sub_dfs = pl.collect_all(
            [
                df.filter(
                    (pl.col("indicator_type") == pair[0])
                    & (pl.col("indicator") == pair[1])
                ).sort(df.columns)
                for pair in synonyms
            ]
        )
        assert all(sub_dfs[0].equals(sub_df) for sub_df in sub_dfs[1:]), (
            "Provided (indicator_type, indicator) pairs are not synonymous."
        )
        synonyms.pop(0)
        df = df.filter(
            [
                (pl.col("indicator_type") != pair[0]) | (pl.col("indicator") != pair[1])
                for pair in synonyms
            ]
        )

    return df


def clean_vaccine(
    df: pl.LazyFrame, colname: str, domain_phrases: Optional[List[str]] = None
) -> pl.LazyFrame:
    """
    Vaccine is the target pathogen plus any formulation information.
    Move extraneous information about eligibilty, etc. to the 'domain'
    column as necessary by specifying the extraneous phrases
    """
    df.rename({colname: "vaccine"})
    df = df.with_columns(pl.col("vaccine").str.to_lowercase())
    if domain_phrases is not None:
        for phrase in domain_phrases:
            df = df.with_columns(
                domain=pl.when(pl.col("vaccine").str.contains(phrase))
                .then(pl.col("domain") + phrase)
                .otherwise(pl.col("domain")),
                vaccine=pl.col("vaccine").str.replace("phrase", ""),
            )
    df = df.with_columns(pl.col("vaccine").str.strip_chars())

    return df


def clean_time_type(df: pl.LazyFrame, time_type: str) -> pl.LazyFrame:
    """
    Time type is the interval between report dates, e.g. 'month' or 'week'
    This is specified directly, as it is only in the data description.
    """
    df = df.with_columns(time_type=pl.lit(time_type))

    return df


def clean_time_start_end(
    df: pl.LazyFrame,
    column: str | List[str],
    col_format: str = "end",
    time_format: str = "%Y-%m-%dT%H:%M:%S%.f",
    time_type: str = "week",
) -> pl.LazyFrame:
    """
    Time start is the date on which phone surveys began for the reported estimate.
    Time end is the date on which phone surveys ended for the reported estimate.
    A list of columns may be given if month-day is in one column and year in another.
    Column format is "start", "end", or "both" depending on which times are given.
    Time format is the date format string describing the format of the (first) column.
    Time type is the interval between report dates, e.g. 'month' or 'week'
    """
    if not isinstance(column, list):
        column = [column]
    if col_format == "end":
        df = df.with_columns(
            time_end=pl.col(column[0])
            .str.strptime(pl.Datetime, time_format)
            .dt.truncate("1d")
        )
        if time_type == "week":
            df = df.with_columns(time_start=pl.col("time_end").dt.offset_by("-6d"))
        elif time_type == "month":
            df = df.with_columns(time_start=pl.col("time_end").dt.offset_by("-1mo"))
        else:
            raise RuntimeError("Time type {time_type} is not recognized.")
    elif col_format == "both":
        df = df.with_columns(
            time_start=pl.col(column[0]).str.extract(r"^(.*?)-").str.strip_chars(),
            time_end=pl.col(column[0]).str.extract(r"^(.*?)-").str.strip_chars(),
        )
        if len(column) > 1:
            df = df.with_columns(
                time_start=(pl.col("time_start") + " " + pl.col(column[1])),
                time_end=(pl.col("time_end") + " " + pl.col(column[1])),
            )
        df = df.with_columns(
            pl.col("time_start")
            .str.strptime(pl.Datetime, time_format)
            .dt.truncate("1d"),
            pl.col("time_end").str.strptime(pl.Datetime, time_format).dt.truncate("1d"),
        )
    else:
        raise RuntimeError("Column format {col_format} is not recognized.")

    return df


def clean_estimate(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """
    Estimate is the percentage of respondents represented by a row.
    """
    df.rename({column: "estimate"})
    df.with_columns(
        (pl.col("estimate").cast(pl.Float64) / 100.0).clip(
            lower_bound=0.0, upper_bound=1.0
        )
    )

    return df


def clean_lci_uci(
    df: pl.LazyFrame, column: str, col_format: str = "end"
) -> pl.LazyFrame:
    """
    LCI and UCI are the lower & upper 95% confidence intervals on the estimate.
    A list of columns may be given if month-day is in one column and year in another.
    Column format is "half" or "full" depending on whether the CI half-width or
    full range is given.
    """
    if col_format == "half":
        df = (
            df.with_columns(pl.col(column).cast(pl.Float64))
            .with_columns(
                lci=(pl.col("estimate") - pl.col(column)).clip(lower_bound=0.0),
                uci=(pl.col("estimate") + pl.col(column)).clip(upper_bound=0.0),
            )
            .drop(column)
        )
    elif col_format == "full":
        df = df.with_columns(
            lci=(
                pl.col(column)
                .str.extract(r"^(.*?)-")
                .str.strip_chars()
                .cast(pl.Float64)
            )
            / 100.0,
            uci=(
                pl.col(column)
                .str.extract(r"-(.*)", 1)
                .str.strip_chars()
                .cast(pl.Float64)
            )
            / 100.0,
        ).drop(column)
    else:
        raise RuntimeError("Column format {col_format} is not recognized.")

    return df


def clean_sample_size(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
    df.rename({column: "sample_size"})
    df = df.with_columns(pl.col("sample_size").cast(pl.UInt32))

    return df


def remove_duplicates(df: pl.LazyFrame, tolerance: float = 0.001) -> pl.LazyFrame:
    """
    Rows are duplicates if they are within some tolerance for value columns
    (estimate, lci, & uci) and identical for group columns (all others).
    To find duplicates, group by group columns and get mean of value columns.
    Then verify that the difference between the raw and mean values < tolerance.
    If duplicate rows are found, average their values together.
    If duplicate groups have clashing values, raise an error.
    """
    value_columns = {"estimate", "lci", "uci"}
    group_columns = data_schema.keys() - value_columns

    bad_groups = (
        df.group_by(group_columns)
        .agg(pl.col(value_columns).pipe(_mean_max_diff, tolerance=tolerance))
        .filter(pl.all_horizontal(value_columns).not_())
        .collect()
    )

    if bad_groups.shape[0] > 0:
        raise RuntimeError("Some identical groups have clashing values:", bad_groups)

    return df.group_by(group_columns).agg(pl.col(value_columns).mean())


def _mean_max_diff(x: pl.Expr, tolerance: float) -> pl.Expr:
    return (x - x.mean()).abs().max() < tolerance


def enforce_schema(df: pl.LazyFrame, schema: pl.Schema = data_schema) -> pl.LazyFrame:
    """
    Enforce that the standardized schema is followed. Remove extra columns.
    """
    current_columns = df.collect_schema().names()
    needed_columns = schema.names()
    missing_columns = set(needed_columns) - set(current_columns)
    extra_columns = set(current_columns) - set(needed_columns)
    if missing_columns != set():
        raise RuntimeError("Missing columns:", missing_columns)
    if extra_columns != set():
        warnings.warn("Dropped columns: {extra_columns}")
    return df.select(needed_columns)


def ensure_eager(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, pl.LazyFrame):
        return df.collect()
    else:
        raise RuntimeError(f"Cannot collect object of type {type(df)}")


def duplicated_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Return duplicated rows of an (eager) data frame

    Args:
        df (pl.DataFrame): input df

    Returns:
        pl.DataFrame: duplicated rows only
    """
    return df.filter(df.is_duplicated())


def rows_with_any_null(df: pl.DataFrame) -> pl.DataFrame:
    """Filter a data frame for rows with any null value

    Args:
        df (pl.LazyFrame): data frame

    Returns:
        pl.LazyFrame: rows with any null value
    """
    return df.filter(pl.any_horizontal(pl.all().is_null()))
