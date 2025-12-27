from typing import List, Literal, get_args

import polars as pl

from nisapi.clean.helpers import (
    admin1_values,
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
    data_schema,
    drop_bad_rows,
    duplicated_rows,
    enforce_schema,
    ensure_eager,
    remove_duplicates,
    rows_with_any_null,
)

VALIDATION_MODES = Literal["warn", "error", "ignore"]


def clean_dataset(
    df: pl.LazyFrame, id: str, clean_args: dict, validation_mode: VALIDATION_MODES
) -> pl.DataFrame:
    """Clean a raw dataset, applying dataset-specific cleaning rules

    Args:
        df (pl.DataFrame): raw dataset
        id (str): dataset id
        clean_args (dict): cleaning arguments for the raw data set
        validation_mode: validation mode. See Validate().

    Returns:
        pl.DataFrame: clean dataset
    """

    out = (
        df.pipe(drop_bad_rows, **clean_args["drop_bad_rows"])
        .pipe(clean_geography_type, **clean_args["clean_geography_type"])
        .pipe(clean_geography, **clean_args["clean_geography"])
        .pipe(clean_domain_type, **clean_args["clean_domain_type"])
        .pipe(clean_domain, **clean_args["clean_domain"])
        .pipe(clean_indicator_type, **clean_args["clean_indicator_type"])
        .pipe(clean_indicator, **clean_args["clean_indicator"])
        .pipe(clean_vaccine, **clean_args["clean_vaccine"])
        .pipe(clean_time_type, **clean_args["clean_time_type"])
        .pipe(clean_time_start_end, **clean_args["clean_time_start_end"])
        .pipe(clean_estimate, **clean_args["clean_estimate"])
        .pipe(clean_lci_uci, **clean_args["clean_lci_uci"])
        .pipe(clean_sample_size, **clean_args["clean_sample_size"])
        .pipe(remove_duplicates, **clean_args["remove_duplicates"])
        .pipe(enforce_schema)
    )

    out = out.pipe(ensure_eager)
    Validate(id=id, df=out, mode=validation_mode)
    return out


class Validate:
    def __init__(self, id: str, df: pl.DataFrame | pl.LazyFrame, mode: str):
        """Set up for validation

        Args:
            id (str): dataset ID, used for validation problem messaging
            df (pl.DataFrame | pl.LazyFrame): dataset to validate
            mode (str): validation mode. One of "warn", "error", "ignore". If ignore, do
                nothing on validation problems. If "warn", print a warning message. If
                error, raise an error.
        """
        self.id = id
        self.df = df.pipe(ensure_eager)

        modes = get_args(VALIDATION_MODES)
        if mode not in modes:
            raise RuntimeError(f"Unknown mode {mode}. Must be one of: {modes}.")

        self.mode = mode

        self.validate()

    def validate(self):
        self.problems = self.get_problems(self.df)

        if len(self.problems) > 0 and self.mode != "ignore":
            print("\n".join([f"❌ id={self.id}: {x}" for x in self.problems]))

            if self.mode == "error":
                raise RuntimeError("Validation problems")

    @classmethod
    def get_problems(cls, df: pl.DataFrame):
        problems = []

        # df must have expected column order and types
        if not df.schema == data_schema:
            problems.append(f"Bad schema: {df.schema}. Expected schema: {data_schema}.")

        # no duplicated rows
        if df.is_duplicated().any():
            rows = df.pipe(duplicated_rows).glimpse(return_as_string=True)
            problems.append(f"Duplicated rows: {rows}")

        # no duplicated values
        if df.drop(["estimate", "lci", "uci"]).is_duplicated().any():
            dup_groups = (
                df.drop(["estimate", "lci", "uci"])
                .pipe(duplicated_rows)
                .glimpse(return_as_string=True)
            )
            problems.append(f"Duplicated groups: {dup_groups}")

        # no null values
        null_rows = df.pipe(rows_with_any_null)
        if null_rows.shape[0] > 0:
            problems.append(f"Null values in rows: {null_rows}")

        # whitespace
        for column in ["domain_type", "domain", "indicator_type", "indicator"]:
            problems += cls.validate_whitespace(df, column=column)

        # Vaccine -------------------------------------------------------------
        # `vaccine` must be in a certain set
        problems += cls.validate_vaccine(df, column="vaccine")

        # Geography ---------------------------------------------------------------
        problems += cls.validate_geography(
            df, type_column="geography_type", value_column="geography"
        )

        # Domain --------------------------------------------------------------
        # age groups should have the form "18-49 years" or "65+ years"
        problems += cls.validate_age_groups(df)

        # Indicators --------------------------------------------------------------
        pass

        # Times -------------------------------------------------------------------
        if not df["time_type"].is_in(["week", "month"]).all():
            bad_time_types = set(df["time_type"].unique()) - set(["week", "month"])
            problems.append(
                f"Bad time type: {bad_time_types}. Expected only 'week' or 'month."
            )

        df_with_intervals = df.with_columns(
            intervals=(pl.col("time_end") - pl.col("time_start")).dt.total_days()
        )

        if not df_with_intervals["intervals"].is_in([6, 7, 8, 28, 29, 30, 31]).all():
            bad_intervals = df_with_intervals.filter(
                pl.col("intervals").is_in([6, 7, 8, 28, 29, 30, 31]).not_()
            ).glimpse(return_as_string=True)
            problems.append(
                f"Unusual intervals between time_start and time_end: {bad_intervals}"
            )

        # Metrics -----------------------------------------------------------------
        # estimates and CIs must be proportions
        for col in ["estimate", "lci", "uci"]:
            if not df[col].is_between(0.0, 1.0).all():
                bad_rows = df.filter(pl.col(col).is_between(0.0, 1.0).not_()).glimpse(
                    return_as_string=True
                )
                problems.append(f"`{col}` is not in range 0-1: {bad_rows}")

        # confidence intervals must bracket estimate
        if not ((df["lci"] <= df["estimate"]) & (df["estimate"] <= df["uci"])).all():
            bad_rows = df.filter(
                (pl.col("lci") > pl.col("estimate"))
                | (pl.col("uci") < pl.col("estimate"))
            ).glimpse(return_as_string=True)
            problems.append(f"confidence intervals do not bracket estimate: {bad_rows}")

        # Sample Sizes ------------------------------------------------------------
        # sample sizes must be positive
        if not (df["sample_size"] > 0).all():
            bad_sample_sizes = df.filter(pl.col("sample_size") <= 0).glimpse(
                return_as_string=True
            )
            problems.append(f"Non-positive sample sizes: {bad_sample_sizes}")

        return problems

    @staticmethod
    def validate_vaccine(df: pl.DataFrame, column: str) -> List[str]:
        bad_vaccines = set(df[column].to_list()) - {
            "flu",
            "covid",
            "flu_h1n1",
            "flu_seasonal_or_h1n1",
            "nirsevimab",
            "rsv_maternal",
            "rsv",
        }
        if len(bad_vaccines) > 0:
            return [f"Bad `vaccine` values: {bad_vaccines}"]
        else:
            return []

    @classmethod
    def validate_geography(
        cls, df: pl.DataFrame, type_column: str, value_column: str
    ) -> List[str]:
        errors = []

        # type must be in a certain set
        errors += cls.bad_value_error(
            type_column,
            df[type_column],
            ["nation", "region", "admin1", "substate", "county", "local"],
        )
        # if type is "nation", value must also be "nation"
        bad_nation_values = (
            df.filter(
                pl.col(type_column) == pl.lit("nation"),
                pl.col(value_column) != pl.lit("nation"),
            )
            .get_column(value_column)
            .unique()
            .to_list()
        )
        if len(bad_nation_values) > 0:
            errors.append(f"Bad nation values: {bad_nation_values}")
        # if type is "region", must be of the form "Region 1"
        bad_region_values = (
            df.filter(
                pl.col(type_column) == pl.lit("region"),
                pl.col(value_column).str.contains(r"^Region \d+$").not_(),
            )
            .get_column(value_column)
            .unique()
            .to_list()
        )
        if len(bad_region_values) > 0:
            errors.append(f"Bad region values: {bad_region_values}")
        # if type is "admin1", value must be in a specific list
        bad_admin1_values = (
            df.filter(
                pl.col(type_column) == pl.lit("admin1"),
                pl.col(value_column).is_in(admin1_values).not_(),
            )
            .get_column(value_column)
            .unique()
            .to_list()
        )
        if len(bad_admin1_values) > 0:
            errors.append(f"Bad admin1 values: {bad_admin1_values}")

        bad_county_values = (
            df.filter(
                pl.col(type_column) == pl.lit("county"),
                pl.col(value_column).str.contains(r"^\d{5}$").not_(),
            )
            .get_column(value_column)
            .unique()
            .to_list()
        )
        if len(bad_county_values) > 0:
            errors.append(f"Bad county values: {bad_county_values}")

        # no validation applies to substate

        return errors

    @classmethod
    def validate_whitespace(cls, df: pl.DataFrame, column: str) -> List[str]:
        bad_values = (
            df.select(pl.col(column).unique())
            .filter(pl.col(column).pipe(cls._has_excess_whitespace))
            .to_series()
            .to_list()
        )

        return [f"Bad whitespace in column {column}: '{x}'" for x in bad_values]

    @staticmethod
    def _has_excess_whitespace(x):
        """
        String contains any of:
          - whitespace other than space
          - multiple whitespaces characters in a row
          - whitespace at the start or end of the string
        """
        return (
            x.str.contains(r"[^\S ]")
            | x.str.contains(r"\s{2,}")
            | x.str.contains(r"^\s+")
            | x.str.contains(r"\s+$")
        )

    @classmethod
    def validate_age_groups(cls, df) -> List[str]:
        age_groups = df.filter(pl.col("domain_type") == pl.lit("age"))[
            "domain"
        ].unique()
        invalid_age_groups = age_groups.filter(
            cls.is_valid_age_group(age_groups).not_()
        ).to_list()

        if len(invalid_age_groups) > 0:
            return [f"Invalid age groups: {invalid_age_groups}"]

        return []

    @staticmethod
    def bad_value_error(
        column_name: str, values: pl.Series, expected: List[str]
    ) -> List[str]:
        bad_values = set(values.to_list()) - set(expected)
        if len(bad_values) > 0:
            return [f"Bad values in `{column_name}`: {bad_values}"]
        else:
            return []

    @staticmethod
    def is_valid_age_group(x: pl.Expr) -> pl.Expr:
        """Which elements of an age group Expr/Series are valid?

        Args:
            x (pl.Expr): bool
        """
        regex1 = r"^\d+-\d+ (months|years)$"  # eg "18-49 years"
        regex2 = r"^\d+\+ (years|months)$"  # eg "65+ years" or "6+ months"
        regex3 = r"^\d+ months-\d+ years$"  # eg "6 months-17 years"
        regex4 = r"^\d+-\d+ years \(high risk\)$"  # eg "18-49 years"
        return (
            x.str.contains(regex1)
            | x.str.contains(regex2)
            | x.str.contains(regex3)
            | x.str.contains(regex4)
        )
