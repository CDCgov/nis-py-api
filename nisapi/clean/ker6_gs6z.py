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
        df.pipe(drop_bad_rows, "suppression_flag")
        .pipe(clean_geography_type, "geographic_level")
        .pipe(clean_geography, "geographic_name")
        .pipe(clean_domain_type, "demographic_level")
        .pipe(clean_domain, "demographic_name")
        .pipe(clean_indicator_type, "indicator_label")
        .pipe(clean_indicator, "indicator_category_label")
        .pipe(clean_vaccine, "vaccine")
        .pipe(clean_time_type, None, override="week")
        .pipe(clean_time_start_end, "week_ending", "end", "%Y-%m-%d")
        .pipe(clean_estimate, "estimate")
        .pipe(clean_lci_uci, "ci_half_width_95pct", "half")
        .pipe(clean_sample_size, "unweighted_sample_size")
        .pipe(
            remove_duplicates,
            synonym_columns=("indicator_type", "indicator"),
            synonyms=[
                ("4-level vaccination and intent", "Vaccinated"),
                ("up-to-date", "Yes"),
            ],
        )
        .pipe(enforce_schema)
    )
