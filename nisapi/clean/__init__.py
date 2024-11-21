import polars as pl
import polars.testing
import nisapi.clean.ksfb_ug5d
import nisapi.clean.udsf_9v7b
import nisapi.clean.sw5n_wg2p
from nisapi.clean.helpers import (
    valid_age_groups,
    assert_valid_geography,
    data_schema,
    ensure_eager,
)


def clean_dataset(id: str, df: pl.DataFrame) -> pl.DataFrame:
    """Clean a raw dataset, applying dataset-specific cleaning rules

    Args:
        id (str): dataset ID
        df (pl.DataFrame): raw dataset

    Returns:
        pl.DataFrame: clean dataset
    """

    if id == "udsf-9v7b":
        out = nisapi.clean.udsf_9v7b.clean(df)
    elif id == "sw5n-wg2p":
        out = nisapi.clean.sw5n_wg2p.clean(df)
    elif id == "ksfb-ug5d":
        out = nisapi.clean.ksfb_ug5d.clean(df)
    else:
        raise RuntimeError(f"No cleaning set up for dataset {id}")

    validate(out)
    return out


def validate(df: pl.DataFrame):
    """Validate a clean dataset

    Args:
        df (pl.DataFrame): dataset
    """
    # force collection, to make validations easier
    df = df.pipe(ensure_eager)

    # df must have expected column order and types
    assert df.schema == data_schema

    # no duplicated rows
    polars.testing.assert_frame_equal(df, df.unique(), check_row_order=False)
    assert not df.is_duplicated().any()

    # no duplicated values
    assert not df.drop(["estimate", "ci_half_width_95pct"]).is_duplicated().any()

    # no null values
    assert df.null_count().pipe(sum).item() == 0

    # `vaccine` must be in a certain set
    assert df["vaccine"].is_in(["flu", "covid"]).all()

    # Geography ---------------------------------------------------------------
    assert_valid_geography(df["geographic_type"], df["geographic_value"])

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
