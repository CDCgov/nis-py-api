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
        .pipe(
            clean_geography_type,
            "geography_type",
            replace={
                "states/local areas": "admin1",
                "counties": "local",
                "hhs regions/national": "region",
            },
        )
        .pipe(clean_geography, "geography", replace={"United States": "nation"})
        .pipe(
            clean_domain_type,
            None,
            donor_colname="dimension_type",
            transfer={
                "Age": "Age & Possible Risk",
                "Race and Ethnicity": "Race & Ethnicity",
                "Years": "Location & Age",
            },
        )
        .pipe(
            clean_domain,
            "dimension",
            donor_colname="dimension_type",
            transfer={
                "6 Months - 17 Years": "6 Months-17 Years",
                ">=18 Years": ">=18 Years",
                "18-49 Years": "18-49 Years",
                "50-64 Years": "50-64 Years",
                ">=65 Years": ">=65 Years",
            },
        )
        .pipe(clean_indicator_type, None, override="Received a vaccination")
        .pipe(clean_indicator, None, override="yes")
        .pipe(
            clean_vaccine,
            "vaccine",
            lowercase=False,
            replace={
                "Seasonal Influenza": "flu",
                "Any Influenza Vaccination, Seasonal or H1N1": "flu_seasonal_or_h1n1",
                "Influenza A (H1N1) 2009 Monovalent": "flu_h1n1",
            },
        )
        .pipe(clean_time_type, None, override="month")
        .pipe(clean_time_start_end, ["month", "year_season"], "end", "%m-%Y")
        .pipe(clean_estimate, "coverage_estimate")
        .pipe(clean_lci_uci, "_95_ci", "full", "to")
        .pipe(clean_sample_size, "population_sample_size")
        .pipe(remove_duplicates)
        .pipe(enforce_schema)
    )
