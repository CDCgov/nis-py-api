import polars as pl
from nisapi.clean.helpers import (
    data_schema,
    drop_suppressed_rows,
    rename_indicator_columns,
    set_lowercase,
    cast_types,
    clean_geography,
    remove_duplicate_rows,
    remove_near_duplicates,
    clean_4_level,
)


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    # this particular dataset has a bad column name
    return (
        df.rename({"estimates": "estimate"})
        .pipe(drop_suppressed_rows)
        .pipe(rename_indicator_columns)
        .select(data_schema.names())
        .pipe(set_lowercase)
        .pipe(cast_types)
        .pipe(clean_geography)
        # this dataset has an error: for `demographic_type="overall"`, it has
        # `demographic_value="18+ years"`, but it should be "overall"
        .with_columns(
            demographic_value=pl.when(pl.col("demographic_type") == pl.lit("overall"))
            .then(pl.lit("overall"))
            .otherwise(pl.col("demographic_value"))
        )
        .pipe(remove_duplicate_rows)
        # Find the rows that are *almost* duplicates: there are some rows that
        # have nearly duplicate values
        .pipe(remove_near_duplicates, tolerance=1e-3, n_fold_duplication=2)
        .pipe(clean_4_level)
    )
