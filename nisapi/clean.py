import polars as pl
import polars.testing
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
    clean = clean.filter(pl.col("suppression_flag") == pl.lit("0"))

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
        clean = clean.pipe(remove_near_duplicates, tolerance=1e-3, n_fold_duplicates=2)

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

    Two rows are "near-duplicate" if they have all the same grouping variables
    (i.e., everything other than estimate and CI) and have estimate and CI
    values that are within some tolerance of one another.

    Args:
        df (pl.DataFrame): input data frame
        tolerance (float): greatest difference in `estimate` and `ci_half_width_95pct`
        n_fold_duplication (int, optional): For each set of grouping values,
          there are exactly this number of near-duplicate rows. If None (default),
          do not apply this kind of check.
        value_columns (Sequence[str]): names of the value columns. Defaults to
          `["estimate", "ci_half_width_95pct"]`.
        group_columns (Sequence[str]): names of the grouping columns. If None
          (the default), uses all columns in `df` that are not in `value_columns`.

    Returns:
        pl.DataFrame: _description_
    """
    if group_columns is None:
        group_columns = set(df.columns) - set(value_columns)

    nearly_dup_rows = df.drop(value_columns).is_duplicated()

    if n_fold_duplication is not None:
        # assert that there are N of each of these duplicated groups
        assert (
            df.filter(nearly_dup_rows)
            .select(group_columns)
            .group_by(pl.all())
            .count()["count"]
            == n_fold_duplication
        ).all()

    # assert that the estimates and CIs in these groups are similar to
    # one another, within some small margin
    assert (
        df.filter(nearly_dup_rows)
        .group_by(group_columns)
        .agg((pl.col(value_columns).pipe(lambda x: x.max() - x.min())))
        .select((pl.col(value_columns) < tolerance).all())
        .select(pl.all_horizontal(value_columns))
        .item()
    )

    # drop the nearly-duplicate rows
    return df.filter(nearly_dup_rows.not_())


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
