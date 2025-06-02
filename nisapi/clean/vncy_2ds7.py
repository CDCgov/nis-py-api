import polars as pl

from nisapi.clean.helpers import (
    cast_types,
    clamp_ci,
    clean_geography,
    drop_suppressed_rows,
    enforce_columns,
    hci_to_cis,
    rename_indicator_columns,
    set_lowercase,
    week_ending_to_times,
)


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.pipe(drop_suppressed_rows)
        .with_columns(vaccine=pl.lit("flu"))
        .pipe(rename_indicator_columns)
        .pipe(set_lowercase)
        .pipe(cast_types)
        .pipe(clean_geography)
        .pipe(week_ending_to_times)
        .pipe(hci_to_cis)
        .pipe(clamp_ci)
        .pipe(enforce_columns)
    )
