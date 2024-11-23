import polars as pl
import uuid
import calendar
from nisapi.clean.helpers import admin1_values, drop_suppressed_rows, enforce_columns


def _clean_geography_expr(type_: pl.Expr, value: pl.Expr) -> pl.Expr:
    out_type = (
        pl.when(type_ == pl.lit("National Estimates"))
        .then(pl.lit("nation"))
        .when(type_ == pl.lit("HHS Regional Estimates"))
        .then(pl.lit("region"))
        .when(
            (type_ == pl.lit("Jurisdictional Estimates")) & (value.is_in(admin1_values))
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


def parse_coninf_95(df: pl.DataFrame) -> pl.DataFrame:
    df_split = df.with_columns(
        pl.col("coninf_95")
        .str.split_exact(" - ", 1)
        .struct.rename_fields(["lci", "uci"])
    )

    return df_split.unnest("coninf_95")


def month_name_to_number(x: pl.Expr) -> pl.Expr:
    # note that we need to do this union because "May" is both a full name and an abbreviation,
    # and replace_strict wants unique old values. Note also that range(13) includes 0, and
    # month_name[0] is ""
    mapping = dict(zip(calendar.month_name, range(13))) | dict(
        zip(calendar.month_abbr, range(13))
    )
    return x.replace_strict(mapping)


def _parse_time_period_expr(time_year: pl.Expr, time_period: pl.Expr) -> pl.Expr:
    year = time_year.cast(pl.Int32)
    period_split = time_period.str.extract_groups(
        r"^(\w+)\s+(\w+)\s+-\s+(\w+)\s+(\w+)\s*$"
    ).struct.rename_fields(["month1", "day1", "month2", "day2"])

    month1 = period_split.struct["month1"].pipe(month_name_to_number)
    day1 = period_split.struct["day1"].str.strip_chars().cast(pl.Int32)
    month2 = period_split.struct["month2"].pipe(month_name_to_number)
    day2 = period_split.struct["day2"].str.strip_chars().cast(pl.Int32)

    date1 = pl.date(year, month1, day1)
    date2 = pl.date(year, month2, day2)

    return pl.struct(time_start=date1, time_end=date2)


def parse_time_period(df: pl.DataFrame) -> pl.DataFrame:
    column_name = str(uuid.uuid1())
    return (
        df.with_columns(
            _parse_time_period_expr(pl.col("time_year"), pl.col("time_period")).alias(
                column_name
            )
        )
        .unnest(column_name)
        .drop(["time_year", "time_period"])
    )


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
        .with_columns(vaccine=pl.lit("covid"))
        .pipe(drop_suppressed_rows)
        .drop("sample_size")
        .pipe(parse_coninf_95)
        .with_columns(pl.col(["estimate", "lci", "uci"]).cast(pl.Float64) / 100)
        .pipe(parse_time_period)
        .with_columns(
            pl.col("time_type").replace_strict({"Monthly": "month", "Weekly": "week"})
        )
        .with_columns(pl.col("demographic_value").pipe(clean_age_group))
        .pipe(clean_geography)
        .pipe(enforce_columns)
    )
