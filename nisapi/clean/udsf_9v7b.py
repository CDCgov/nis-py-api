import polars as pl
import uuid
import nisapi.clean


def _clean_geography_expr(type_: pl.Expr, value: pl.Expr) -> pl.Expr:
    out_type = (
        pl.when(type_ == pl.lit("National Estimates"))
        .then(pl.lit("nation"))
        .when(type_ == pl.lit("HHS Regional Estimates"))
        .then(pl.lit("region"))
        .when(
            (type_ == pl.lit("Jurisdictional Estimates"))
            & (value.is_in(nisapi.clean.admin1_values))
        )
        .then(pl.lit("admin1"))
        .when(
            (type_ == pl.lit("Jurisdictional Estimates"))
            & (value.str.contains(r"^[A-Z]{2}-"))
        )
        .then(pl.lit("substate"))
    )

    out_value = (
        pl.when(out_type == pl.lit("nation"))
        .then(pl.lit("nation"))
        .when(out_type == pl.lit("region"))
        .then(value.pipe(clean_region))
        .when(out_type.is_in(["admin1", "substate"]))
        .then(value)
    )

    return pl.struct(geographic_type=out_type, geographic_value=out_value)


def clean_geography(df: pl.DataFrame) -> pl.DataFrame:
    geography_column = str(uuid.uuid1())
    return (
        df.with_columns(
            _clean_geography_expr(
                pl.col("geographic_type"), pl.col("geographic_value")
            ).alias(geography_column)
        )
        .drop(["geographic_type", "geographic_value"])
        .unnest(geography_column)
    )


def clean_age_group(x: pl.Expr) -> pl.Expr:
    return x.str.replace(" – ", "-")


def clean_region(x: pl.Expr) -> pl.Expr:
    return x.str.extract("^(Region \\d+): ", 1)


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.rename(
            {
                "geography_type": "geographic_type",
                "geography": "geographic_value",
                "group_name": "demographic_type",
                "group_category": "demographic_value",
                "indicator_name": "indicator_type",
                "indicator_category": "indicator_value",
            }
        )
        .pipe(nisapi.clean.drop_suppressed_rows)
        .drop("sample_size")
        .pipe(nisapi.clean.parse_coninf_95)
        .pipe(nisapi.clean.parse_time_period)
        .with_columns(
            pl.col("time_type").replace_strict({"Monthly": "month", "Weekly": "week"})
        )
        .with_columns(pl.col(["estimate", "lci", "uci"]).cast(pl.Float64))
        .with_columns(pl.col("demographic_value").pipe(clean_age_group))
        .pipe(clean_geography)
    )
