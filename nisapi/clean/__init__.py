import polars as pl
import polars.testing

import nisapi.clean.ksfb_ug5d
import nisapi.clean.sw5n_wg2p
import nisapi.clean.udsf_9v7b
import nisapi.clean.vh55_3he6
from nisapi.clean.helpers import (
    admin1_values,
    data_schema,
    duplicated_rows,
    ensure_eager,
    rows_with_any_null,
)


def clean_dataset(df: pl.DataFrame, id: str) -> pl.DataFrame:
    """Clean a raw dataset, applying dataset-specific cleaning rules

    Args:
        df (pl.DataFrame): raw dataset
        id (str): dataset ID

    Returns:
        pl.DataFrame: clean dataset
    """

    if id == "udsf-9v7b":
        out = nisapi.clean.udsf_9v7b.clean(df)
    elif id == "sw5n-wg2p":
        out = nisapi.clean.sw5n_wg2p.clean(df)
    elif id == "ksfb-ug5d":
        out = nisapi.clean.ksfb_ug5d.clean(df)
    elif id == "vh55-3he6":
        out = nisapi.clean.vh55_3he6.clean(df)
    else:
        raise RuntimeError(f"No cleaning set up for dataset {id}")

    out = out.pipe(ensure_eager)
    Validate(id=id, df=out)
    return out


class Validate:
    def __init__(self, id: str, df: pl.DataFrame | pl.LazyFrame):
        self.id = id
        self.df = df.pipe(ensure_eager)
        self.validate()

    def validate(self):
        self.errors = self.get_validation_errors(self.df)
        if len(self.errors) > 0:
            print(f"Validation errors in dataset ID: {self.id}")
            print(*self.errors, sep="\n")

            raise RuntimeError("Validation errors")

    @classmethod
    def get_validation_errors(cls, df: pl.DataFrame):
        errors = []
        warnings = []

        # df must have expected column order and types
        if not df.schema == data_schema:
            errors.append(f"Bad schema: {df.schema}")

        # no duplicated rows
        if df.is_duplicated().any():
            rows = df.pipe(duplicated_rows).glimpse(return_as_string=True)
            errors.append(f"Duplicated rows: {rows}")

        # no duplicated values
        if df.drop(["estimate", "lci", "uci"]).is_duplicated().any():
            dup_groups = (
                df.drop(["estimate", "lci", "uci"])
                .pipe(duplicated_rows)
                .glimpse(return_as_string=True)
            )
            errors.append(f"Duplicated groups: {dup_groups}")

        # no null values
        if df.null_count().pipe(sum).item() > 0:
            counts = df.null_count()
            null_columns = counts.select(
                col for col in counts.columns if (counts[col] > 0).any()
            )
            null_rows = df.pipe(rows_with_any_null)
            errors.append(f"Null values: {null_columns} {null_rows}")

        # Vaccine -------------------------------------------------------------
        # `vaccine` must be in a certain set
        errors += cls.validate_vaccine(df, column="vaccine")

        # Geography ---------------------------------------------------------------
        errors += cls.validate_geography(
            df, type_column="geography_type", value_column="geography"
        )

        # domains ------------------------------------------------------------
        # age groups should have the form "18-49 years" or "65+ years"
        age_groups = df.filter(pl.col("domain_type") == pl.lit("age"))[
            "domain"
        ].unique()
        invalid_age_groups = age_groups.filter(
            cls.is_valid_age_group(age_groups).not_()
        ).to_list()
        if len(invalid_age_groups) > 0:
            errors.append(f"Invalid age groups: {invalid_age_groups}")

        # Indicators --------------------------------------------------------------
        pass

        # Times -------------------------------------------------------------------
        if not df["time_type"].is_in(["week", "month"]).all():
            errors.append("Bad time type")

        if not (df["time_start"] <= df["time_end"]).all():
            errors.append("Not all time starts are before time ends")

        # Metrics -----------------------------------------------------------------
        # estimates and CIs must be proportions
        if not df["estimate"].is_between(0.0, 1.0).all():
            bad_rows = df.filter(pl.col("estimate").is_between(0.0, 1.0).not_())
            errors.append(f"`Estimate` is not in range 0-1: {bad_rows}")
        for col in ["lci", "uci"]:
            if not df[col].is_between(0.0, 1.0).all():
                bad_rows = df.filter(pl.col(col).is_between(0.0, 1.0).not_())
                warnings.append(f"`{col}` is not in range 0-1: {bad_rows}")

        # confidence intervals must bracket estimate
        if not ((df["lci"] <= df["estimate"]) & (df["estimate"] <= df["uci"])).all():
            errors.append("confidence intervals do not bracket estimate")

        return errors

    @staticmethod
    def validate_vaccine(df: pl.DataFrame, column: str) -> [str]:
        bad_vaccines = set(df[column].to_list()) - {
            "flu",
            "covid",
            "flu_h1n1",
            "flu_seasonal_or_h1n1",
        }
        if len(bad_vaccines) > 0:
            return [f"Bad `vaccine` values: {bad_vaccines}"]
        else:
            return []

    @classmethod
    def validate_geography(
        cls, df: pl.DataFrame, type_column: str, value_column: str
    ) -> [str]:
        errors = []

        # type must be in a certain set
        errors += cls.bad_value_error(
            type_column,
            df[type_column],
            ["nation", "region", "admin1", "substate", "county"],
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

    @staticmethod
    def bad_value_error(column_name: str, values: pl.Series, expected: [str]) -> [str]:
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
        regex1 = r"^\d+-\d+ years$"  # eg "18-49 years"
        regex2 = r"^\d+\+ (years|months)$"  # eg "65+ years" or "6+ months"
        regex3 = r"^\d+ months-\d+ years$"  # eg "6 months-17 years"
        return x.str.contains(regex1) | x.str.contains(regex2) | x.str.contains(regex3)
