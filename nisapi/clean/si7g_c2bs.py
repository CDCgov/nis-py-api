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
        .pipe(clean_geography_type, "geography_type")
        .pipe(clean_geography, "geography")
        .pipe(clean_domain_type, "group_name")
        .pipe(
            clean_domain,
            "group_category",
            donor_colname="new_vax_group",
            transfer={
                "(among adults age 60-74 with high-risk conditions)": "60-74 years, high risk",
                "(among adults age 75+)": "75+ years, high risk",
            },
        )
        .pipe(clean_indicator_type, "indicator_name")
        .pipe(clean_indicator, "indicator_category")
        .pipe(
            clean_vaccine,
            "new_vax_group",
            replace={
                "covid-19": "covid",
                "rsv (among adults age 60-74 with high-risk conditions)": "rsv",
                "rsv (among adults age 75+)": "rsv",
            },
        )
        .pipe(clean_time_type, "time_type")
        .pipe(clean_time_start_end, ["time_period", "year"], "both", "%B %d %Y")
        .pipe(clean_estimate, "estimate")
        .pipe(clean_lci_uci, "_95_ci", "full")
        .pipe(clean_sample_size, "sample_size")
        .pipe(remove_duplicates)
        .pipe(enforce_schema)
    )
