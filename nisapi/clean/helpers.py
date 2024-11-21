import polars as pl
import uuid
from typing import Sequence

"""Data schema to be used for all datasets"""
data_schema = pl.Schema(
    [
        ("vaccine", pl.String),
        ("geographic_type", pl.String),
        ("geographic_value", pl.String),
        ("demographic_type", pl.String),
        ("demographic_value", pl.String),
        ("indicator_type", pl.String),
        ("indicator_value", pl.String),
        ("week_ending", pl.Date),
        ("estimate", pl.Float64),
        ("ci_half_width_95pct", pl.Float64),
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
        .select((pl.col("indicator_value") == pl.lit("yes")).all())
        .pipe(ensure_eager)
        .item()
    )

    # check that "Yes" and "Received a vaccination" are the same thing, so that
    # we can drop "Up to Date"
    assert (
        df.filter(pl.col("indicator_value").is_in(["yes", "received a vaccination"]))
        .drop("indicator_type")
        .pipe(ensure_eager)
        .pivot(on="indicator_value", values=["estimate", "ci_half_width_95pct"])
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
                "geographic_type",
                "demographic_type",
                "indicator_value",
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
        pl.col("geographic_type").replace({"national": "nation"}),
        pl.col("geographic_value").replace({"National": "nation"}),
    ).with_columns(
        pl.col("geographic_type").replace_strict(
            {
                "nation": "nation",
                "state": "admin1",
                "region": "region",
                "substate": "substate",
            }
        )
    )


def remove_duplicate_rows(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.filter(df.is_duplicated().not_())


def rename_indicator_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Make "indicator" follow the same logic as "geography" and
    "demographic", with "type" and "value" columns
    """
    return df.rename(
        {
            "geographic_level": "geographic_type",
            "geographic_name": "geographic_value",
            "demographic_level": "demographic_type",
            "demographic_name": "demographic_value",
            "indicator_label": "indicator_type",
            "indicator_category_label": "indicator_value",
        }
    )


def remove_near_duplicates(
    df: pl.DataFrame,
    tolerance: float,
    n_fold_duplication: int = None,
    value_columns: Sequence[str] = ["estimate", "ci_half_width_95pct"],
    group_columns: Sequence[str] = None,
) -> pl.DataFrame:
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
    if group_columns is None:
        group_columns = set(df.columns) - set(value_columns)

    assert set(group_columns).issubset(df.columns)
    assert set(value_columns).issubset(df.columns)

    if n_fold_duplication is not None:
        # ensure we have a group size column without collisions
        group_size_col = str(uuid.uuid1())
        assert group_size_col not in group_columns

        # all groups that aren't of size 1 should be the "fold" duplication size
        assert (
            df.group_by(group_columns)
            .len(name=group_size_col)
            .filter(pl.col(group_size_col) > 1)[group_size_col]
            == n_fold_duplication
        ).all()

    # check that the difference between summarized values and input values is
    # less than the tolerance
    out_spread = df.group_by(group_columns).agg(
        pl.col(value_columns).pipe(_mean_max_diff, tolerance=tolerance)
    )
    out_spread_bad = out_spread.filter(pl.all_horizontal(value_columns).not_())
    if out_spread_bad.shape[0] > 0:
        raise RuntimeError("Some groups violate tolerance:", out_spread_bad)

    return df.group_by(group_columns).agg(pl.col(value_columns).mean())


def _mean_max_diff(x: pl.Expr, tolerance: float) -> pl.Expr:
    return (x - x.mean()).abs().max() < tolerance


def drop_suppressed_rows(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("suppression_flag") == pl.lit("0")).drop("suppression_flag")


def ensure_eager(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, pl.LazyFrame):
        return df.collect()
    else:
        raise RuntimeError(f"Cannot collect object of type {type(df)}")


def assert_valid_geography(type_: pl.Series, value: pl.Series) -> None:
    # type must be in a certain set
    assert type_.is_in(["nation", "region", "admin1", "substate"]).all()
    # if type is "nation", value must also be "nation"
    assert (value.filter(type_ == "nation") == "nation").all()
    # if type is "region", must be of the form "Region 1"
    assert value.filter(type_ == "region").str.contains(r"^Region \d+$").all()
    # if type is "admin1", value must be in a specific list
    assert value.filter(type_ == "admin1").is_in(admin1_values).all()
    # no validation applies to substate


def valid_age_groups(x: pl.Series) -> bool:
    """Validate that a series of age groups is valid

    Args:
        x (pl.Series): series of age groups
    """
    return x.str.contains(r"^(\d+-\d+|\d+\+) years$").all()
