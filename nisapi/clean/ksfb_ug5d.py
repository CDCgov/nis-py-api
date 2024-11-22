import polars as pl
from nisapi.clean.helpers import (
    drop_suppressed_rows,
    rename_indicator_columns,
    set_lowercase,
    cast_types,
    clean_geography,
    clean_4_level,
    remove_near_duplicates,
    replace_overall_demographic_value,
    week_ending_to_times,
    hci_to_cis,
    enforce_columns,
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
        .pipe(week_ending_to_times)
        .pipe(hci_to_cis)
        .pipe(enforce_columns)
        .pipe(replace_overall_demographic_value)
    )
