import polars as pl
from nisapi.clean.helpers import (
    data_schema,
    drop_suppressed_rows,
    rename_indicator_columns,
    set_lowercase,
    cast_types,
    clean_geography,
    clean_4_level,
    remove_near_duplicates,
)


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.pipe(drop_suppressed_rows)
        .pipe(rename_indicator_columns)
        .pipe(set_lowercase)
        .pipe(cast_types)
        .pipe(clean_geography)
        .unique()
        .pipe(remove_near_duplicates, tolerance=1e-3, n_fold_duplication=2)
        .pipe(clean_4_level)
        .select(data_schema.names())
    )
