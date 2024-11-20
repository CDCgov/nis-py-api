import polars as pl
import polars.testing
from typing import Sequence
import uuid

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


def clean_dataset(id: str, df: pl.DataFrame) -> pl.DataFrame:
    """Clean a raw dataset, applying dataset-specific cleaning rules

    Args:
        id (str): dataset ID
        df (pl.DataFrame): raw dataset

    Returns:
        pl.DataFrame: clean dataset
    """
    clean = df

    if id not in ["sw5n-wg2p", "ksfb-ug5d"]:
        raise RuntimeError(f"No cleaning set up for dataset {id}")

    # Drop rows with suppression flags
    clean = clean.pipe(drop_suppressed_rows)

    if id == "sw5n-wg2p":
        # this particular dataset has a bad column name
        clean = clean.rename({"estimates": "estimate"})

    if id in ["sw5n-wg2p", "ksfb-ug5d"]:
        clean = (
            clean.pipe(rename_indicator_columns)
            # drop unneeded columns
            .select(data_schema.names())
            # change most string columns to all lowercase
            # (but don't change, e.g., the names of states)
            .with_columns(
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

        clean = (
            clean
            # Change from "national" to "nation", so that types are nouns rather
            # than adjectives. (Otherwise we would need to change "region" to "regional")
            .with_columns(
                pl.col("geographic_type").replace({"national": "nation"}),
                pl.col("geographic_value").replace({"National": "nation"}),
            )
            # "sw5n-wg2p" only:
            # this dataset has an error: for `demographic_type="overall"`, it has
            # `demographic_value="18+ years"`, but it should be "overall"
            .with_columns(
                demographic_value=pl.when(
                    pl.col("demographic_type") == pl.lit("overall")
                )
                .then(pl.lit("overall"))
                .otherwise(pl.col("demographic_value"))
            )
        )

        # there should now be no nulls in any column
        assert clean.null_count().pipe(sum).item() == 0

        # Remove duplicate rows
        clean = clean.filter(clean.is_duplicated().not_())

        # Find the rows that are *almost* duplicates: there are some rows that
        # have nearly duplicate values
        clean = clean.pipe(remove_near_duplicates, tolerance=1e-3, n_fold_duplication=2)

        # Verify that indicator type "up-to-date" has only one value ("yes")
        assert clean.filter(pl.col("indicator_type") == pl.lit("up-to-date")).pipe(
            col_values_in, "indicator_value", ["yes"]
        )

        # check that "Yes" and "Received a vaccination" are the same thing, so that
        # we can drop "Up to Date"
        assert (
            clean.filter(
                pl.col("indicator_value").is_in(["yes", "received a vaccination"])
            )
            .drop("indicator_type")
            .pivot(on="indicator_value", values=["estimate", "ci_half_width_95pct"])
            .select(
                (
                    (
                        pl.col("estimate_yes")
                        == pl.col("estimate_received a vaccination")
                    )
                    & (
                        pl.col("ci_half_width_95pct_yes")
                        == pl.col("ci_half_width_95pct_received a vaccination")
                    )
                ).all()
            )
            .item()
        )

        clean = clean.filter(
            pl.col("indicator_type") == pl.lit("4-level vaccination and intent")
        )

        return clean

    validate(clean)
    return clean


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


def col_values_in(df: pl.DataFrame, col: str, values: str) -> bool:
    """All values of `df` in column `col` are in `values`?"""
    return df[col].is_in(values).all()


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


def validate(df: pl.DataFrame):
    """Validate a clean dataset

    Args:
        df (pl.DataFrame): dataset
    """
    # force collection, to make validations easier
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # df must have expected column order and types
    assert df.schema == data_schema

    # no duplicated rows
    polars.testing.assert_frame_equal(df, df.unique(), check_row_order=False)

    # no null values in crucial columns
    for col in df.columns:
        assert not df[col].is_null().any()

    # `vaccine` must be in a certain set
    assert df["vaccine"].is_in(["flu", "covid"]).all()

    # Geography ---------------------------------------------------------------
    # `geographic_type` must be in a certain set
    assert df["geographic_type"].is_in(["nation", "region", "state", "substate"]).all()
    # if `geographic_type` is "nation", `geographic_value` must also be "nation"
    assert (
        df.filter(pl.col("geographic_type") == pl.lit("nation"))["geographic_value"]
        == "nation"
    ).all()

    # Demographics ------------------------------------------------------------
    # if `demographic_type` is "overall", `demographic_value` must also be "overall"
    assert (
        df.filter(pl.col("demographic_type") == pl.lit("overall"))["demographic_value"]
        == "overall"
    ).all()
    # age groups should have the form "18-49 years" or "65+ years"
    assert valid_age_groups(
        df.filter(pl.col("demographic_type") == pl.lit("age"))["demographic_value"]
    )

    # Indicators --------------------------------------------------------------
    assert df["indicator_type"].is_in(["4-level vaccination and intent"]).all()

    # Metrics -----------------------------------------------------------------
    # estimates must be percents
    assert df["estimate"].is_between(0.0, 1.0).all()
    # confidence intervals must be non-negative
    assert (df["ci_half_width_95pct"] >= 0.0).all()


def valid_age_groups(x: pl.Series) -> bool:
    """Validate that a series of age groups is valid

    Args:
        x (pl.Series): series of age groups
    """
    return x.str.contains(r"^(\d+-\d+|\d+\+) years$").all()
