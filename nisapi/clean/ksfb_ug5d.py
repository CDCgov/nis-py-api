import polars as pl
from nisapi.clean.helpers import (
    data_schema,
    drop_suppressed_rows,
    rename_indicator_columns,
    set_lowercase,
    cast_types,
    clean_geography,
    remove_duplicate_rows,
    clean_4_level,
)


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.pipe(drop_suppressed_rows)
        .pipe(rename_indicator_columns)
        .select(data_schema.names())
        .pipe(set_lowercase)
        .pipe(cast_types)
        .pipe(clean_geography)
        .pipe(remove_duplicate_rows)
        .pipe(clean_4_level)
    )
