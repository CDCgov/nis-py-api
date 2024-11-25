import polars as pl
import polars.testing
import nisapi.clean.ksfb_ug5d
import nisapi.clean.udsf_9v7b
import nisapi.clean.sw5n_wg2p
from nisapi.clean.helpers import (
    is_valid_age_groups,
    is_valid_geography,
    data_schema,
    ensure_eager,
    duplicated_rows,
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

    @staticmethod
    def get_validation_errors(df: pl.DataFrame):
        errors = []

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
            null_rows = df.pipe(rows_with_any_null)
            errors.append(f"Null values: {null_rows}")

        # Vaccine -------------------------------------------------------------
        # `vaccine` must be in a certain set
        if not df["vaccine"].is_in(["flu", "covid"]).all():
            errors.append("Bad `vaccine` values")

        # Geography ---------------------------------------------------------------
        if not is_valid_geography(df["geographic_type"], df["geographic_value"]):
            errors.append("Invalid geography")

        # Demographics ------------------------------------------------------------
        # if `demographic_type` is "overall", `demographic_value` must also be "overall"
        overall_demographic_values = (
            df.filter(pl.col("demographic_type") == pl.lit("overall"))[
                "demographic_value"
            ]
            .unique()
            .to_list()
        )
        if overall_demographic_values != ["overall"]:
            errors.append(
                f"Bad overall demographic values: {overall_demographic_values}"
            )
        # age groups should have the form "18-49 years" or "65+ years"
        if not is_valid_age_groups(
            df.filter(pl.col("demographic_type") == pl.lit("age"))["demographic_value"]
        ):
            errors.append("Invalid age groups")

        # Indicators --------------------------------------------------------------
        pass

        # Times -------------------------------------------------------------------
        if not df["time_type"].is_in(["week", "month"]).all():
            errors.append("Bad time type")

        if not (df["time_start"] <= df["time_end"]).all():
            errors.append("Not all time starts are before time ends")

        # Metrics -----------------------------------------------------------------
        # estimates and CIs must be proportions
        for col in ["estimate", "lci", "uci"]:
            if not df[col].is_between(0.0, 1.0).all():
                bad_rows = df.filter(pl.col(col).is_between(0.0, 1.0).not_())
                errors.append(f"`{col}` is not in range 0-1: {bad_rows}")

        # confidence intervals must bracket estimate
        if not ((df["lci"] <= df["estimate"]) & (df["estimate"] <= df["uci"])).all():
            errors.append("confidence intervals do not bracket estimate")

        return errors
