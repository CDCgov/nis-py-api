import polars as pl

from nisapi.clean.helpers import (
    clamp_ci,
    clean_geography,
    drop_suppressed_rows,
    enforce_columns,
    remove_duplicate_rows,
    replace_overall_domain,
    set_lowercase,
)


def rename_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.rename(
        {
            "group_name": "domain_type",
            "group_category": "domain",
            "indicator_name": "indicator_type",
            "indicator_category": "indicator",
            "new_vax_group": "vaccine",
        }
    )


def cast_numeric_types(df: pl.LazyFrame) -> pl.LazyFrame:
    out = (
        df.with_columns(
            pl.col("estimate").cast(pl.Float64),
            lci=pl.col("_95_ci")
            .str.extract(r"^(.*?)-")
            .str.strip_chars()
            .cast(pl.Float64),
            uci=pl.col("_95_ci")
            .str.extract(r"-(.*)", 1)
            .str.strip_chars()
            .cast(pl.Float64),
        )
        .with_columns(
            pl.col("estimate") / 100.0,
            pl.col("lci") / 100.0,
            pl.col("uci") / 100.0,
        )
        .with_columns(ci_half_width_95pct=(pl.col("uci") - pl.col("estimate")))
        .drop("_95_ci")
    )

    return out


def cast_date_types(df: pl.LazyFrame) -> pl.LazyFrame:
    out = (
        df.with_columns(
            time_start=pl.col("time_period").str.extract(r"^(.*?)-").str.strip_chars(),
            time_end=pl.col("time_period").str.extract(r"^(.*?)-").str.strip_chars(),
            time_type=pl.lit("month"),
        )
        .with_columns(
            time_start=(pl.col("time_start") + " " + pl.col("year")),
            time_end=(pl.col("time_end") + " " + pl.col("year")),
        )
        .with_columns(
            pl.col("time_start").str.strptime(pl.Datetime, "%B %d %Y").dt.date(),
            pl.col("time_end").str.strptime(pl.Datetime, "%B %d %Y").dt.date(),
        )
        .drop("year")
    )

    return out


def clean_vaccine_names(df: pl.LazyFrame) -> pl.LazyFrame:
    out = (
        df.with_columns(
            domain=pl.when(
                pl.col("vaccine")
                == "rsv (among adults age 60-74 with high-risk conditions)"
            )
            .then(
                pl.col("domain") + "(among adults age 60-74 with high-risk conditions)"
            )
            .otherwise(pl.col("domain")),
        )
        .with_columns(
            domain=pl.when(pl.col("vaccine") == "rsv (among adults age 75+)")
            .then(pl.col("domain") + "(among adults age 75+)")
            .otherwise(pl.col("domain")),
        )
        .with_columns(
            vaccine=pl.col("vaccine").replace(
                {
                    "covid-19": "covid",
                    "rsv (among adults age 60-74 with high-risk conditions)": "rsv",
                    "rsv (among adults age 75+)": "rsv",
                }
            )
        )
    )

    return out


def clean_regions(df: pl.LazyFrame) -> pl.LazyFrame:
    out = df.with_columns(
        geography=pl.when(pl.col("geography_type") == "region")
        .then(
            pl.col("geography")
            .str.extract(r"^(.*?):")
            .str.to_lowercase()
            .str.to_titlecase()
        )
        .otherwise(pl.col("geography"))
    )

    return out


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.pipe(drop_suppressed_rows)
        .pipe(rename_columns)
        .pipe(set_lowercase)
        .pipe(cast_numeric_types)
        .pipe(cast_date_types)
        .pipe(clean_geography)
        .pipe(clean_regions)
        .pipe(clean_vaccine_names)
        .pipe(clamp_ci)
        .pipe(enforce_columns)
        .pipe(remove_duplicate_rows)
        .pipe(replace_overall_domain)
    )
