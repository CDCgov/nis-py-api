from typing import List

import polars as pl

import nisapi.clean.akkj_j5ru
import nisapi.clean.k4cb_dxd7
import nisapi.clean.ker6_gs6z
import nisapi.clean.ksfb_ug5d
import nisapi.clean.sw5n_wg2p
import nisapi.clean.vdz4_qrri
import nisapi.clean.vh55_3he6
import nisapi.clean.vncy_2ds7
from nisapi.clean.helpers import (
    admin1_values,
    data_schema,
    duplicated_rows,
    ensure_eager,
    rows_with_any_null,
)


def clean_dataset(df: pl.LazyFrame, id: str, validation_mode: str) -> pl.DataFrame:
    """Clean a raw dataset, applying dataset-specific cleaning rules

    Args:
        df (pl.DataFrame): raw dataset
        id (str): dataset ID
        validation_mode (str): validation mode

    Returns:
        pl.DataFrame: clean dataset
    """

    if id == "akkj-j5ru":
        out = nisapi.clean.akkj_j5ru.clean(df)
    elif id == "sw5n-wg2p":
        out = nisapi.clean.sw5n_wg2p.clean(df)
    elif id == "ksfb-ug5d":
        out = nisapi.clean.ksfb_ug5d.clean(df)
    elif id == "vh55-3he6":
        out = nisapi.clean.vh55_3he6.clean(df)
    elif id == "vdz4-qrri":
        out = nisapi.clean.vdz4_qrri.clean(df)
    elif id == "ker6-gs6z":
        out = nisapi.clean.ker6_gs6z.clean(df)
    elif id == "vncy-2ds7":
        out = nisapi.clean.vncy_2ds7.clean(df)
    elif id == "k4cb-dxd7":
        out = nisapi.clean.k4cb_dxd7.clean(df)
    else:
        raise RuntimeError(f"No cleaning set up for dataset {id}")

    out = out.pipe(ensure_eager)
    Validate(id=id, df=out, mode=validation_mode)
    return out


class Validate:
    modes = ["warn", "error", "ignore"]

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

        if mode not in self.modes:
            raise RuntimeError(f"Unknown mode {mode}. Must be one of: {self.modes}.")

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
            problems.append(f"Bad schema: {df.schema}")

        # no duplicated rows
        if df.is_duplicated().any():
            rows = df.pipe(duplicated_rows).glimpse(return_as_string=True)
            problems.append(f"Duplicated rows: {rows}")

        # no duplicated values
        if not {"estimate", "lci", "uci"}.issubset(df.columns):
            problems.append("Missing columns: `estimate`, `lci`, or `uci`")
        elif df.drop(["estimate", "lci", "uci"]).is_duplicated().any():
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

        # Vaccine -------------------------------------------------------------
        # `vaccine` must be in a certain set
        problems += cls.validate_vaccine(df, column="vaccine")

        # Geography ---------------------------------------------------------------
        problems += cls.validate_geography(
            df, type_column="geography_type", value_column="geography"
        )

        # domains ------------------------------------------------------------
        # age groups should have the form "18-49 years" or "65+ years"
        problems += cls.validate_age_groups(df)

        # Indicators --------------------------------------------------------------
        pass

        # Times -------------------------------------------------------------------
        if "time_type" not in df.columns:
            problems.append("Missing column `time_type`")
        elif not df["time_type"].is_in(["week", "month"]).all():
            problems.append("Bad time type")

        if not {"time_start", "time_end"}.issubset(df.columns):
            problems.append("Missing columns `time_start` or `time_end`")
        elif not (df["time_start"] <= df["time_end"]).all():
            problems.append("Not all time starts are before time ends")

        # Metrics -----------------------------------------------------------------
        # estimates and CIs must be proportions
        for col in ["estimate", "lci", "uci"]:
            if not df[col].is_between(0.0, 1.0).all():
                bad_rows = df.filter(pl.col(col).is_between(0.0, 1.0).not_())
                problems.append(f"`{col}` is not in range 0-1: {bad_rows}")

        # confidence intervals must bracket estimate
        if not ((df["lci"] <= df["estimate"]) & (df["estimate"] <= df["uci"])).all():
            problems.append("confidence intervals do not bracket estimate")

        return problems

    @staticmethod
    def validate_vaccine(df: pl.DataFrame, column: str) -> List[str]:
        if column not in df.columns:
            return [f"Missing column `{column}`"]

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
        if not {type_column, value_column}.issubset(df.columns):
            return [f"Missing columns `{type_column}` or `{value_column}`"]

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

    @classmethod
    def validate_age_groups(cls, df) -> List[str]:
        if "domain_type" not in df.columns:
            return ["Missing column `domain_type`"]

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
        regex1 = r"^\d+-\d+ years$"  # eg "18-49 years"
        regex2 = r"^\d+\+ (years|months)$"  # eg "65+ years" or "6+ months"
        regex3 = r"^\d+ months-\d+ years$"  # eg "6 months-17 years"
        regex4 = r"^\d+-\d+ months$"  # eg "6-23 months"
        return (
            x.str.contains(regex1)
            | x.str.contains(regex2)
            | x.str.contains(regex3)
            | x.str.contains(regex4)
        )
