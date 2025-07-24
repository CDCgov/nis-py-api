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
        df.pipe(drop_bad_rows, None)
        .pipe(clean_geography_type, "geography_type")
        .pipe(clean_geography, "geography")
        .pipe(
            clean_domain, "dimension", "dimension_type", ["Age", "Race and Ethnicity"]
        )  # Note domain before domain_type here
        .pipe(
            clean_domain_type,
            None,
            override="Age, Risk, Race and Ethnicity, and/or Location",
        )
        .pipe(clean_indicator_type, None, override="received a vaccination")
        .pipe(clean_indicator, None, override="yes")
        .pipe(clean_vaccine, "vaccine")
        .pipe(clean_time_type, None, "month")
        .pipe(clean_time_start_end, ["month", "year_season"], "end", "%m-%Y")
        .pipe(clean_estimate, "coverage_estimate")
        .pipe(clean_lci_uci, "_95_ci", "full")
        .pipe(clean_sample_size, "unweighted_sample_size")
        .pipe(remove_duplicates)
        .pipe(enforce_schema)
    )
