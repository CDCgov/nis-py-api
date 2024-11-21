import polars as pl
from nisapi.clean.helpers import (
    data_schema,
    drop_suppressed_rows,
    rename_indicator_columns,
    set_lowercase,
    cast_types,
    clean_geography,
    remove_near_duplicates,
    clean_4_level,
    replace_overall_demographic_value,
)


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    # this particular dataset has a bad column name
    return (
        df.rename({"estimates": "estimate"})
        .pipe(drop_suppressed_rows)
        .pipe(rename_indicator_columns)
        .unique()
        .pipe(set_lowercase)
        .pipe(cast_types)
        .pipe(clean_geography)
        .pipe(replace_overall_demographic_value)
        # Find the rows that are *almost* duplicates: there are some rows that
        # have nearly duplicate values
        .pipe(remove_near_duplicates, tolerance=1e-3, n_fold_duplication=2)
        .pipe(clean_4_level)
        .select(data_schema.names())
    )
