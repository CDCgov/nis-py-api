import polars as pl
from .helpers import admin1_values


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


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df.rename({"geography": "geography_name"})
        .pipe(
            clean_geography,
            type_column="geography_type",
            name_column="geography_name",
            fips_column="fips",
        )
        .pipe(clean_time, year_season_column="year_season", month_column="month")
    )
