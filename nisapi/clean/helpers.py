import uuid
import warnings
from typing import Iterable, List, Optional, Sequence

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


def clean_indicator(df: pl.LazyFrame, colname: str) -> pl.LazyFrame:
    """
    Indicator is the specific answer to the survey question.
    """
    df.rename({colname: "indicator"})
    df = df.with_columns(pl.col("indicator").str.strip_chars())

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
            raise ValueError("Time type {time_type} is not recognized.")
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
        raise ValueError("Column format {col_format} is not recognized.")

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
        raise ValueError("Column format {col_format} is not recognized.")

    return df


def clean_sample_size(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
    df.rename({column: "sample_size"})
    df = df.with_columns(pl.col("sample_size").cast(pl.UInt32))

    return df


def clean_4_level(df: pl.LazyFrame) -> pl.LazyFrame:
    # Verify that indicator type "up-to-date" has only one value ("yes")
    assert (
        df.filter(pl.col("indicator_type") == pl.lit("up-to-date"))
        .select((pl.col("indicator") == pl.lit("yes")).all())
        .pipe(ensure_eager)
        .item()
    )

    # check that "Yes" and "Received a vaccination" are the same thing, so that
    # we can drop "Up to Date"
    assert (
        df.filter(pl.col("indicator").is_in(["yes", "received a vaccination"]))
        .drop("indicator_type")
        .pipe(ensure_eager)
        .pivot(on="indicator", values=["estimate", "ci_half_width_95pct"])
        .select(
            (
                (pl.col("estimate_yes") == pl.col("estimate_received a vaccination"))
                & (
                    pl.col("ci_half_width_95pct_yes")
                    == pl.col("ci_half_width_95pct_received a vaccination")
                )
            ).all()
        )
        .item()
    )

    return df.filter(
        pl.col("indicator_type") == pl.lit("4-level vaccination and intent")
    )


def remove_duplicate_rows(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.filter(df.collect().is_duplicated().not_())


def remove_near_duplicates(
    df: pl.LazyFrame,
    tolerance: float,
    n_fold_duplication: Optional[int] = None,
    value_columns: Sequence[str] = ["estimate", "ci_half_width_95pct"],
    group_columns: Optional[Iterable[str]] = None,
) -> pl.LazyFrame:
    """Remove near-duplicate rows

    Rows are "near-duplicate" if they have the same grouping variables
    (e.g., everything other than estimate and CI) and have estimate and CI
    values that are within some tolerance of one another.

    This function removes duplicates by
      1. Grouping by group columns
      2. Summarizing the value columns using the mean value
      3. Checking that the difference between the raw values and the
        summarized values is below some tolerance

    Args:
        df (pl.DataFrame): Input data frame
        tolerance (float): Greatest permissible difference between summarized
          value and input value
        n_fold_duplication (int, optional): For each set of grouping values,
          assert there are exactly this number of near-duplicate rows. If
          None (default), do not do this check.
        value_columns (Sequence[str]): names of the value columns. Defaults to
          `["estimate", "ci_half_width_95pct"]`.
        group_columns (Sequence[str]): names of the grouping columns. If None
          (the default), uses all columns in `df` that are not in `value_columns`.

    Returns:
        pl.DataFrame: data frame with columns `group_columns` and `value_columns`
          and at most as many rows as in `df`
    """
    columns = df.collect_schema().names()

    if group_columns is None:
        group_columns = set(columns) - set(value_columns)

    assert set(group_columns).issubset(columns)
    assert set(value_columns).issubset(columns)

    if n_fold_duplication is not None:
        # ensure we have a group size column without collisions
        group_size_col = str(uuid.uuid1())
        assert group_size_col not in group_columns

        # all groups that aren't of size 1 should be the "fold" duplication size
        assert (
            df.group_by(group_columns)
            .len(name=group_size_col)
            .filter(pl.col(group_size_col) > 1)
            .select((pl.col(group_size_col) == n_fold_duplication).all())
            .pipe(ensure_eager)
            .item()
        )

    # check that the difference between summarized values and input values is
    # less than the tolerance
    out_spread = df.group_by(group_columns).agg(
        pl.col(value_columns).pipe(_mean_max_diff, tolerance=tolerance)
    )
    out_spread_bad = out_spread.filter(pl.all_horizontal(value_columns).not_()).pipe(
        ensure_eager
    )
    if out_spread_bad.shape[0] > 0:
        raise RuntimeError("Some groups violate tolerance:", out_spread_bad)

    return df.group_by(group_columns).agg(pl.col(value_columns).mean())


def _mean_max_diff(x: pl.Expr, tolerance: float) -> pl.Expr:
    return (x - x.mean()).abs().max() < tolerance


def ensure_eager(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, pl.LazyFrame):
        return df.collect()
    else:
        raise RuntimeError(f"Cannot collect object of type {type(df)}")


def enforce_columns(df: pl.LazyFrame, schema: pl.Schema = data_schema) -> pl.LazyFrame:
    """Enforce columns from the data schema

    Check that input data frame has all the needed columns, then select only those
    column

    Args:
        df (pl.LazyFrame): input data frame
        schema (pl.Schema): data schema. Defaults to `nisapi.clean.helpers.data_schema`.

    Returns:
        pl.LazyFrame: `df`, but with only the columns in the data schema
    """
    # Include a clause to remove leading/trailing whitespace, str.strip_chars, from all string cols
    # Include a message about which columns were removed
    current_columns = df.collect_schema().names()
    needed_columns = schema.names()
    missing_columns = set(needed_columns) - set(current_columns)
    if missing_columns != set():
        raise RuntimeError("Missing columns:", missing_columns)
    return df.select(needed_columns)


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


def clamp_ci(
    df: pl.LazyFrame, lci: pl.Expr = pl.col("lci"), uci: pl.Expr = pl.col("uci")
) -> pl.LazyFrame:
    """Clamp the confidence intervals `lci` and `uci` to be within [0, 1]"""
    return df.with_columns(lci.clip(lower_bound=0.0), uci.clip(upper_bound=1.0))
