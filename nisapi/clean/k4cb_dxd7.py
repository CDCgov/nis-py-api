import polars as pl

from nisapi.clean.helpers import (
    cast_types,
    clamp_ci,
    clean_4_level,
    clean_geography,
    drop_suppressed_rows,
    enforce_columns,
    hci_to_cis,
    replace_overall_domain,
    set_lowercase,
    week_ending_to_times,
)


def rename_indicator_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Make "indicator" follow the same logic as "geography" and
    "domain", with "type" and "value" columns
    """
    return df.rename(
        {
            "geography_level": "geography_type",
            "geography_name": "geography",
            "demographic_level": "domain_type",
            "demographic_name": "domain",
            "indicator_label": "indicator_type",
            "indicator_category_label": "indicator",
        }
    )


def merge_age_groups(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Make the "age_group" column part of the "domain" and "domain_type" columns.
    """
    return (
        df.with_columns(
            domain_type=("age & " + pl.col("domain_type")),
            domain=pl.concat_str(["age_group", "domain"], separator=" & "),
        )
        .with_columns(
            domain_type=pl.when(pl.col("domain_type") == "age & age")
            .then(pl.lit("age"))
            .otherwise(pl.col("domain_type")),
            domain=pl.when(pl.col("domain_type") == "age & age")
            .then(pl.col("age_group"))
            .otherwise(pl.col("domain")),
        )
        .drop("age_group")
    )


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.rename({"suppresion_flag": "suppression_flag"})
        .pipe(drop_suppressed_rows)
        .pipe(rename_indicator_columns)
        .pipe(set_lowercase)
        .pipe(cast_types)
        .pipe(clean_geography)
        .pipe(clean_4_level)
        .pipe(replace_overall_domain)
        .pipe(merge_age_groups)
        .pipe(week_ending_to_times)
        .pipe(hci_to_cis)
        .pipe(clamp_ci)
        .pipe(enforce_columns)
    )
