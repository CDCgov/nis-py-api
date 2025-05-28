import polars as pl

import nisapi.clean.helpers


def clean(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        df
        # get the indicators we need
        .with_columns(
            vaccine=pl.col("indicator_category_label").replace(
                {
                    "Infant received nirsevimab": "nirsevimab",
                    "Mother received RSV vaccination during pregnancy and infant did not receive nirsevimab": "rsv_maternal",
                    # same as above, but with two "the"s
                    "The mother received RSV vaccination during pregnancy and the infant did not receive nirsevimab": "rsv_maternal",
                }
            )
        )
        .filter(pl.col("vaccine").is_in(["nirsevimab", "rsv_maternal"]))
        # separate CI and timeframe columns
        .with_columns(
            pl.col("_95_confidence_interval")
            .str.split_exact(by=" - ", n=2)
            .struct.rename_fields(["lci", "uci"]),
            pl.col("timeframe")
            .str.split_exact(by="-", n=2)
            .struct.rename_fields(["time_start", "time_end"]),
        )
        .select(
            [
                pl.col("timeframe").struct.unnest(),
                "vaccine",
                "estimate",
                pl.col("_95_confidence_interval").struct.unnest(),
            ]
        )
        # adjust column types
        .with_columns(
            pl.col(["estimate", "lci", "uci"]).cast(pl.Float64) / 100.0,
            pl.col(["time_start", "time_end"]).str.to_date(format="%m/%d/%Y"),
        )
        # move to standard schema
        .with_columns(
            geography_type=pl.lit("nation"),
            geography=pl.lit("nation"),
            domain_type=pl.lit("age and season"),
            domain=pl.lit(
                "adult females aged 18-49 years with infants under the age of 8 months during the RSV season (born since April 1, 2024)"
            ),
            indicator_type=pl.lit("received immunization"),
            indicator=pl.lit("received immunization"),
            time_type=pl.lit("month"),
        )
        .select(nisapi.clean.helpers.data_schema.names())
    )
