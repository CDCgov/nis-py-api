import polars as pl

from nisapi.clean.helpers import (
    clamp_ci,
    clean_geography,
    drop_suppressed_rows,
    enforce_columns,
    ensure_eager,
    hci_to_cis,
    rename_indicator_columns,
    set_lowercase,
    week_ending_to_times,
)


def cast_types(df: pl.LazyFrame) -> pl.LazyFrame:
    # This function is specific to ker6-gs6z, which as no times attached
    # to the week_ending dates
    out = df.with_columns(
        pl.col("week_ending").str.strptime(pl.Datetime, "%Y-%m-%d"),
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


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.pipe(drop_suppressed_rows)
        .pipe(rename_indicator_columns)
        .pipe(set_lowercase)
        .pipe(cast_types)
        .pipe(clean_geography)
        .pipe(week_ending_to_times)
        .pipe(hci_to_cis)
        .pipe(clamp_ci)
        .drop(["unweighted_sample_size", "month_week"])
        .pipe(enforce_columns)
    )
