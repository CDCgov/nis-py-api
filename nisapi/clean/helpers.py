import uuid
from typing import Iterable, Optional, Sequence

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


def set_lowercase(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        pl.col(
            [
                "vaccine",
                "geography_type",
                "domain_type",
                "indicator",
                "indicator_type",
            ]
        ).str.to_lowercase()
    )


def cast_types(df: pl.LazyFrame) -> pl.LazyFrame:
    out = df.with_columns(
        pl.col("week_ending").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f"),
        pl.col(["estimate", "ci_half_width_95pct"]).cast(pl.Float64),
    ).with_columns(pl.col(["estimate", "ci_half_width_95pct"]) / 100.0)

    # check that the date doesn't have any trailing seconds
    assert (
        out.select(
            (pl.col("week_ending").dt.truncate("1d") == pl.col("week_ending")).all()
        )
        .pipe(ensure_eager)
        .item()
    )

    return out.with_columns(pl.col("week_ending").dt.date())


def clean_geography(df: pl.LazyFrame) -> pl.LazyFrame:
    # Change from "national" to "nation", so that types are nouns rather
    # than adjectives. (Otherwise we would need to change "region" to "regional")
    return df.with_columns(
        pl.col("geography_type").replace({"national": "nation"}),
        pl.col("geography").replace({"National": "nation"}),
    ).with_columns(
        pl.col("geography_type").replace_strict(
            {
                "nation": "nation",
                "state": "admin1",
                "region": "region",
                "substate": "substate",
                "local": "local",
                "hhs region": "region",
            }
        )
    )


def remove_duplicate_rows(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.filter(df.collect().is_duplicated().not_())


def rename_indicator_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Make "indicator" follow the same logic as "geography" and
    "domain", with "type" and "value" columns
    """
    return df.rename(
        {
            "geographic_level": "geography_type",
            "geographic_name": "geography",
            "demographic_level": "domain_type",
            "demographic_name": "domain",
            "indicator_label": "indicator_type",
            "indicator_category_label": "indicator",
        }
    )


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


def replace_overall_domain(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(pl.col("domain_type").replace({"overall": "age"}))


def _mean_max_diff(x: pl.Expr, tolerance: float) -> pl.Expr:
    return (x - x.mean()).abs().max() < tolerance


def drop_suppressed_rows(df: pl.LazyFrame) -> pl.LazyFrame:
    """Drop rows with suppression flag `"0"`"""
    return df.filter(pl.col("suppression_flag") == pl.lit("0")).drop("suppression_flag")


def ensure_eager(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, pl.LazyFrame):
        return df.collect()
    else:
        raise RuntimeError(f"Cannot collect object of type {type(df)}")


def week_ending_to_times(df: pl.LazyFrame) -> pl.LazyFrame:
    """Convert `week_ending` to time type, time start, and time end

    Args:
        df (pl.LazyFrame): input frame with column "week_ending"

    Returns:
        pl.LazyFrame: frame without "week_ending" but with "time_type",
          "time_start", and "time_end"
    """
    name = str(uuid.uuid1())
    return (
        df.with_columns(_week_ending_to_times_expr(pl.col("week_ending")).alias(name))
        .unnest(name)
        .drop("week_ending")
        .with_columns(time_type=pl.lit("week"))
    )


def _week_ending_to_times_expr(week_ending: pl.Expr) -> pl.Expr:
    week_start = week_ending.dt.offset_by("-6d")
    return pl.struct(
        time_start=week_start, time_end=week_ending, time_type=pl.lit("week")
    )


def hci_to_cis(
    df: pl.LazyFrame,
    estimate_name: str = "estimate",
    hci_name: str = "ci_half_width_95pct",
) -> pl.LazyFrame:
    """Convert estimate and half confidence interval columns into lower and upper columns

    Args:
        df (pl.LazyFrame): input data frame
        estimate_name (str): defaults to "estimate"
        hci_name (str): defaults to "ci_half_width_95pct"

    Returns:
        pl.LazyFrame: data frame without `hci_name` column but with `lci` and `uci`
    """
    name = str(uuid.uuid1())
    return (
        df.with_columns(
            _hci_to_cis_expr(pl.col(estimate_name), pl.col(hci_name)).alias(name)
        )
        .unnest(name)
        .drop(hci_name)
    )


def _hci_to_cis_expr(estimate: pl.Expr, hci: pl.Expr) -> pl.Expr:
    """Convert estimate and half confidence interval into lower and upper interval

    Args:
        estimate (pl.Expr): point estimate
        hci (pl.Expr): half width of the confidence interval

    Returns:
        pl.Expr: struct with fields `lci` and `uci`
    """
    return pl.struct(lci=(estimate - hci).clip(lower_bound=0.0), uci=estimate + hci)


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
