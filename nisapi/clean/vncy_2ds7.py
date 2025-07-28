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
        df.pipe(
            drop_bad_rows,
            "suppression_flag",
            [
                "legend_label",
                "indicator_category_label_1",
                "demographic_level_sort_order",
                "geography_name_order",
                "geography_level_sort_order",
                "season_sort",
                "legend_sort",
            ],
        )
        .pipe(clean_geography_type, "geographic_level")
        .pipe(clean_geography, "geographic_name")
        .pipe(clean_domain_type, "demographic_level")
        .pipe(clean_domain, "demographic_name")
        .pipe(clean_indicator_type, "indicator_label")
        .pipe(
            clean_indicator,
            "indicator_category_label",
            synonyms=[
                ("4-level vaccination and intent", "Received a vaccination"),
                ("up-to-date", "Yes"),
            ],
        )
        .pipe(clean_vaccine, None, override="flu")
        .pipe(clean_time_type, None, "week")
        .pipe(clean_time_start_end, "week_ending")
        .pipe(clean_estimate, "estimate")
        .pipe(clean_lci_uci, "ci_half_width_95pct", "half")
        .pipe(clean_sample_size, "unweighted_sample_size")
        .pipe(remove_duplicates)
        .pipe(enforce_schema)
    )
