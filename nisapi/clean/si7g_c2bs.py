import polars as pl

from nisapi.clean.helpers import (
    clamp_ci,
    clean_4_level,
    clean_geography,
    drop_suppressed_rows,
    enforce_columns,
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
    out = df.with_columns(
        pl.col("estimate").cast(pl.Float64),
        lci=pl.col("_95_ci").str.split("-").arr.first().cast(pl.Float64),
        uci=pl.col("_95_ci").str.split("-").arr.last().cast(pl.Float64),
    ).with_columns(
        pl.col("estimate") / 100.0,
        pl.col("lci") / 100.0,
        pl.col("uci") / 100.0,
    )

    return out


def cast_date_types(df: pl.LazyFrame) -> pl.LazyFrame:
    out = (
        df.with_columns(
            time_start=pl.col("time_period")
            .str.split("-")
            .arr.first()
            .str.strip_chars_end(),
            time_end=pl.col("time_period")
            .str.split("-")
            .arr.last()
            .str.strip_chars_end(),
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


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.pipe(drop_suppressed_rows)
        .pipe(rename_columns)
        .pipe(set_lowercase)
        .pipe(cast_numeric_types)
        .pipe(cast_date_types)
        .pipe(clean_geography)
        # .pipe(clean_4_level)
        .pipe(clamp_ci)
        .pipe(enforce_columns)
        .pipe(replace_overall_domain)
    )
