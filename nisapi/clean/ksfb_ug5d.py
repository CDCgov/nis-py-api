import polars as pl

from nisapi.clean.helpers import (
    cast_types,
    clamp_ci,
    clean_4_level,
    clean_geography,
    drop_suppressed_rows,
    enforce_columns,
    hci_to_cis,
    remove_near_duplicates,
    rename_indicator_columns,
    replace_overall_domain,
    set_lowercase,
    week_ending_to_times,
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
        .pipe(clamp_ci)
        .pipe(enforce_columns)
        .pipe(replace_overall_domain)
    )
