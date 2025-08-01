import polars as pl

from nisapi.clean.helpers import (
    clean_domain,
    clean_domain_type,
    clean_estimate,
    clean_geography,
    clean_geography_type,
    clean_indicator,
    clean_indicator_type,
    clean_lci_uci,
    clean_sample_size,
    clean_time_start_end,
    clean_time_type,
    clean_vaccine,
    drop_bad_rows,
    enforce_schema,
    remove_duplicates,
)


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.pipe(drop_bad_rows, "suppressed_flag")
        .pipe(clean_geography_type, "geography_label")
        .pipe(clean_geography, None, override="nation")
        .pipe(clean_domain_type, None, override="age & season")
        .pipe(
            clean_domain,
            None,
            override="adult females aged 18-49 years with infants under the age of 8 months during the RSV season (born since April 1, 2024)",
        )
        .pipe(clean_indicator_type, None, override="4-level vaccination and intent")
        .pipe(clean_indicator, "indicator_category_label")
        .pipe(
            clean_vaccine,
            None,
            donor_colname="indicator",
            transfer={
                "nirsevimab for infant": "nirsevimab",
                "infant nirsevimab": "nirsevimab",
                "Infant received nirsevimab": "nirsevimab",
                "Mother received": "rsv_maternal",
                "The mother received": "rsv_maternal",
                "RSV vaccine": "rsv",
            },
        )
        .pipe(clean_time_type, None, override="month")
        .pipe(clean_time_start_end, "timeframe", "both", "%m/%d/%Y")
        .pipe(clean_estimate, "estimate")
        .pipe(clean_lci_uci, "_95_confidence_interval", "full")
        .pipe(clean_sample_size, "sample_size")
        .pipe(remove_duplicates)
        .pipe(enforce_schema)
    )
