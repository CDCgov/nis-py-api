import uuid

import polars as pl

from .helpers import admin1_values, clamp_ci, enforce_columns


def _clean_geography_expr(type_: pl.Expr, name: pl.Expr, fips: pl.Expr) -> pl.Expr:
    new_type = (
        pl.when(
            (type_ == pl.lit("HHS Regions/National"))
            & (name == pl.lit("United States"))
        )
        .then(pl.lit("nation"))
        .when(
            (type_ == pl.lit("HHS Regions/National"))
            & (name != pl.lit("United States"))
        )
        .then(pl.lit("region"))
        .when((type_ == pl.lit("States/Local Areas")) & (name.is_in(admin1_values)))
        .then(pl.lit("admin1"))
        .when(
            (type_ == pl.lit("States/Local Areas")) & (name.is_in(admin1_values).not_())
        )
        .then(pl.lit("substate"))
        .when(type_ == pl.lit("Counties"))
        .then(pl.lit("county"))
    )

    new_name = (
        pl.when(new_type == pl.lit("nation"))
        .then(pl.lit("nation"))
        .when(new_type.is_in(["region", "admin1", "substate"]))
        .then(name)
        .when(new_type == pl.lit("county"))
        .then(fips)
    )

    return pl.struct(type_=new_type, name=new_name)


def clean_geography(
    df: pl.LazyFrame, type_column: str, name_column: str, fips_column: str
) -> pl.LazyFrame:
    new_geography = _clean_geography_expr(
        type_=pl.col(type_column), name=pl.col(name_column), fips=pl.col(fips_column)
    )
    return df.with_columns(
        new_geography.struct[0].alias(type_column),
        new_geography.struct[1].alias(name_column),
    ).drop(fips_column)


def clean_time(
    df: pl.LazyFrame, year_season_column: str, month_column: str
) -> pl.LazyFrame:
    """Clean time columns

    If geography is a county, then year/season is a year, and month is trivial.
    Otherwise, year/season is a season, and all months except June (6) are present.
    All seasons are of the form 20XX-YY, where YY=XX+1.

    Args:
        df (pl.LazyFrame): data frame
        year_season_column (str): column name for year/season
        month_column (str): column name for month

    Returns:
        pl.LazyFrame: year/season and month columns are dropped, columns
            time_start and time_end are added
    """

    new_time = _clean_time_expr(pl.col(year_season_column), pl.col(month_column))
    return df.with_columns(
        new_time.struct[0].alias("time_start"), new_time.struct[1].alias("time_end")
    ).drop([year_season_column, month_column])


def _clean_time_expr(year_season: pl.Expr, month: pl.Expr) -> pl.Expr:
    time_start = (
        pl.when(year_season.str.contains(r"^\d{4}$"))
        .then(
            pl.date(
                year=year_season.str.slice(0, 4).cast(pl.Int32),
                month=month.cast(pl.Int32),
                day=1,
            )
        )
        .when(year_season.str.contains(r"^\d{4}-\d{2}$"))
        .then(_clean_time_season_expr(year_season, month))
    )

    time_end = time_start.dt.month_end()

    return pl.struct(time_start=time_start, time_end=time_end)


def _clean_time_season_expr(season: pl.Expr, month: pl.Expr) -> pl.Expr:
    year1 = season.str.slice(0, 4).cast(pl.Int32)
    year2 = year1 + 1
    month = month.cast(pl.Int32)

    year = pl.when(month < 6).then(year2).when(month > 6).then(year1)

    return pl.date(year=year, month=month, day=1)


def clean_domain_indicator(
    df: pl.LazyFrame, type_column: str, value_column: str
) -> pl.LazyFrame:
    new_column_name = str(uuid.uuid1())

    new_column = _clean_domain_indicator_expr(pl.col(type_column), pl.col(value_column))
    return (
        df.with_columns(new_column.alias(new_column_name))
        .drop([type_column, value_column])
        .unnest(new_column_name)
    )


def _clean_domain_indicator_expr(type_: pl.Expr, value: pl.Expr) -> pl.Expr:
    place_age_groups = [
        "6 Months - 17 Years",
        ">=18 Years",
        "18-49 Years",
        "18-64 Years",
        "50-64 Years",
        ">=65 Years",
    ]

    # there are three kinds of "dimension_type": age groups (which signal that
    # "dimension" is place of vaccination), the word "Age", and the phrase
    # "Race and Ethnicity"
    group = (
        pl.when(
            (type_ == pl.lit("Age"))
            & (value.str.to_lowercase().str.contains("(risk|target)"))
        )
        .then(pl.lit("age_risk"))
        .when(type_ == pl.lit("Age"))
        .then(pl.lit("age"))
        .when(type_ == pl.lit("Race and Ethnicity"))
        .then(pl.lit("race/ethnicity"))
        .when(type_.is_in(place_age_groups))
        .then(pl.lit("place"))
    )

    domain_type = group.replace({"place": "age"})

    domain = (
        pl.when(group == pl.lit("place"))
        .then(_clean_age(type_))
        .when(group.is_in(["age", "age_risk"]))
        .then(_clean_age(value))
        .when(group == "race/ethnicity")
        .then(value)
    )

    indicator_type = (
        pl.when(group == pl.lit("place"))
        .then(pl.lit("uptake at place of vaccination"))
        .otherwise(pl.lit("uptake"))
    )

    indicator = (
        pl.when(group == pl.lit("place"))
        .then(value)
        .otherwise(pl.lit("received a vaccination"))
    )

    return pl.struct(
        domain_type=domain_type,
        domain=domain,
        indicator_type=indicator_type,
        indicator=indicator,
    )


def _clean_age(x: pl.Expr) -> pl.Expr:
    return (
        x.str.to_lowercase()
        .str.replace(r">=(\d+)", "$1+")
        .str.replace(r" - ", "-")
        .replace(
            {
                "greater 65": "65+ years",
                "greater than 18 years flu": "18+ years",
                "greater than 6 months flu": "6+ months",
                "at high risk (initial target group)": "at high risk",
                "not in initial target group": "not at high risk",
            }
        )
    )


def clean_estimate(x: pl.Expr) -> pl.Expr:
    return pl.when(x == pl.lit("NR †"))


def clean_ci(
    df: pl.LazyFrame, ci_column: str, lci_clip: float = None, uci_clip: float = None
) -> pl.LazyFrame:
    ci = _clean_ci_expr(pl.col(ci_column))
    return df.with_columns(ci.struct[0].alias("lci"), ci.struct[1].alias("uci")).drop(
        ci_column
    )


def _clean_ci_expr(x: pl.Expr) -> pl.Expr:
    return (
        x.str.replace(r" â€¡$", "")
        .str.replace(r" ‡$", "")
        .str.extract_groups(r"^(-?\d+\.\d+) to (\d+\.\d+)$")
        .struct.rename_fields(["lci", "uci"])
        .struct.with_fields(
            pl.field("lci").cast(pl.Float64) / 100.0,
            pl.field("uci").cast(pl.Float64) / 100.0,
        )
    )


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df
        # rename vaccines
        .with_columns(
            pl.col("vaccine").replace_strict(
                {
                    "Seasonal Influenza": "flu",
                    "Any Influenza Vaccination, Seasonal or H1N1": "flu_seasonal_or_h1n1",
                    "Influenza A (H1N1) 2009 Monovalent": "flu_h1n1",
                }
            )
        )
        .pipe(
            clean_geography,
            type_column="geography_type",
            name_column="geography",
            fips_column="fips",
        )
        .with_columns(pl.lit("month").alias("time_type"))
        .pipe(clean_time, year_season_column="year_season", month_column="month")
        .pipe(
            clean_domain_indicator,
            type_column="dimension_type",
            value_column="dimension",
        )
        # drop unreported estimates; convert to proportion
        .rename({"coverage_estimate": "estimate"})
        .filter(pl.col("estimate").str.starts_with("NR").not_())
        .with_columns(pl.col("estimate").cast(pl.Float64) / 100)
        .pipe(clean_ci, ci_column="_95_ci", lci_clip=0.0, uci_clip=1.0)
        .pipe(clamp_ci)
        .pipe(enforce_columns)
    )
