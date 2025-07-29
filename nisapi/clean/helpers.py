import warnings
from typing import List, Optional, Tuple

import polars as pl

"""Data schema to be used for all datasets"""
data_schema = pl.Schema(
    [
        ("vaccine", pl.String),
        ("geography_type", pl.String),
        ("geography", pl.String),
        ("domain_type", pl.String),
        ("domain", pl.String),
        ("indicator_type", pl.String),
        ("indicator", pl.String),
        ("time_type", pl.String),
        ("time_start", pl.Date),
        ("time_end", pl.Date),
        ("estimate", pl.Float64),
        ("lci", pl.Float64),
        ("uci", pl.Float64),
        ("sample_size", pl.UInt32),
    ]
)

"""First-level administrative divisions of the US: states, territories, and DC"""
admin1_values = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
    "District of Columbia",
    "Guam",
    "Puerto Rico",
    "U.S. Virgin Islands",
]


def drop_bad_rows(
    df: pl.LazyFrame, colname: str | None, bad_columns: Optional[str | List[str]] = None
) -> pl.LazyFrame:
    """
    Bad rows are those with a suppression flag or null values.
    Bad columns are those that should be dropped immediately,
    and where null values are not a problem.
    """
    if colname is not None:
        df = df.filter(pl.col(colname) == pl.lit("0")).drop(colname)
    if bad_columns is not None:
        df = df.drop(bad_columns)
    null_rows = df.filter(pl.any_horizontal(pl.all().is_null())).collect()
    if null_rows.shape[0] > 0:
        warnings.warn("Some rows contain null values. These rows will be dropped.")
        print(null_rows)
    df = df.drop_nulls()

    return df


def clean_geography_type(
    df: pl.LazyFrame,
    colname: str | None,
    override: Optional[str] = None,
    lowercase: bool = True,
    replace: Optional[dict] = None,
    append: Optional[str | List[str]] = None,
    infer: Optional[dict] = None,
    donor_colname: Optional[str] = None,
    transfer: Optional[str | list[str]] = None,
) -> pl.LazyFrame:
    """
    Geography type is the scale of geographic division.
    Add to the default 'replace' dictonary as necessary to standardize verbiage.
    """
    if replace is None:
        replace = {
            "national": "nation",
            "nation": "nation",
            "national estimates": "nation",
            "state": "admin1",
            "state/local areas": "admin1",
            "jurisdictional estimates": "admin1",
            "region": "region",
            "hhs region": "region",
            "hhs regions/national": "region",
            "substate": "substate",
            "local": "local",
            "counties": "local",
        }
    df = (
        df.pipe(_replace_column_name, "geography_type", colname, override)
        .pipe(
            _replace_column_values, "geography_type", lowercase, replace, append, infer
        )
        .pipe(_borrow_column_values, "geography_type", donor_colname, transfer)
    )

    return df


def clean_geography(
    df: pl.LazyFrame,
    colname: str | None,
    override: Optional[str] = None,
    lowercase: bool = False,
    replace: Optional[dict] = None,
    append: Optional[str | List[str]] = None,
    infer: Optional[dict] = None,
    donor_colname: Optional[str] = None,
    transfer: Optional[str | list[str]] = None,
) -> pl.LazyFrame:
    """
    Geography is the specific geographic location.
    Add to the default `replace` dictionary as necessary to standardize verbiage.
    """
    if replace is None:
        replace = {"National": "nation"}
    df = (
        df.pipe(_replace_column_name, "geography", colname, override)
        .pipe(_replace_column_values, "geography", lowercase, replace, append, infer)
        .pipe(_borrow_column_values, "geography", donor_colname, transfer)
    )
    df = df.with_columns(
        geography=pl.when(
            (pl.col("geography_type") == "region")
            & (pl.col("geography").str.contains(":"))
        )
        .then(pl.col("geography").str.extract(r"^(.*?):").str.to_titlecase())
        .otherwise(pl.col("geography")),
        geography_type=pl.when(
            (pl.col("geography_type") == "admin1")
            & (~pl.col("geography").is_in(admin1_values))
        )
        .then(pl.lit("substate"))
        .when(
            (pl.col("geography_type") == "region") & (pl.col("geography") == "nation")
        )
        .then(pl.lit("nation"))
        .otherwise(pl.col("geography_type")),
    )

    return df


def clean_domain_type(
    df: pl.LazyFrame,
    colname: str | None,
    override: Optional[str] = None,
    lowercase: bool = True,
    replace: Optional[dict] = None,
    append: Optional[str | List[str]] = None,
    infer: Optional[dict] = None,
    donor_colname: Optional[str] = None,
    transfer: Optional[str | list[str]] = None,
) -> pl.LazyFrame:
    """
    Domain type is the demographic feature used to define groups.
    Add to the default 'replace' dictionary as necessary to standardize verbiage.
    Another column (e.g. 'age_group') may contain further domain info;
    in this case, provide name(s) for this extra type info (e.g. 'age').
    An override domain type can also be given to fill in all rows.
    """
    if replace is None:
        replace = {"overall": "age"}
    df = (
        df.pipe(_replace_column_name, "domain_type", colname, override)
        .pipe(_replace_column_values, "domain_type", lowercase, replace, append, infer)
        .pipe(_borrow_column_values, "domain_type", donor_colname, transfer)
    )
    # if extra_type is not None:
    #     if not isinstance(extra_type, list):
    #         extra_type = [extra_type]
    #     df = df.with_columns(
    #         domain_type=pl.when(pl.col("domain_type").is_in(extra_type))
    #         .then(pl.col("domain_type"))
    #         .otherwise(pl.col("domain_type") + " & " + " & ".join(extra_type))
    #     )

    return df


def clean_domain(
    df: pl.LazyFrame,
    colname: str | None,
    override: Optional[str] = None,
    lowercase: bool = False,
    replace: Optional[dict] = None,
    append: Optional[str | List[str]] = None,
    infer: Optional[dict] = None,
    donor_colname: Optional[str] = None,
    transfer: Optional[str | list[str]] = None,
) -> pl.LazyFrame:
    """
    Domain is the specific demographic group.
    Add to the `replace` dictionary as necessary to standardize verbiage.
    Another column (e.g. 'age_group') may contain further domain info;
    in this case, provide this column's existing name and preferred
    name(s) for this extra type info (e.g. 'age').
    An override domain can also be given to fill in all rows.
    """
    if replace is None:
        replace = {"All adults 18+": "18+ years", "Overall": "6 months-17 years"}
    df = (
        df.pipe(_replace_column_name, "domain", colname, override)
        .pipe(_replace_column_values, "domain", lowercase, replace, append, infer)
        .pipe(_borrow_column_values, "domain", donor_colname, transfer)
    )
    # if extra_column is not None and extra_type is not None:
    #     if not isinstance(extra_type, list):
    #         extra_type = [extra_type]
    #     df = df.with_columns(
    #         domain=pl.when(pl.col("domain_type").is_in(extra_type))
    #         .then(pl.col("domain"))
    #         .otherwise(pl.concat_str(["domain", extra_column], separator=" & "))
    #     ).drop(extra_column)

    return df


def clean_indicator_type(
    df: pl.LazyFrame,
    colname: str | None,
    override: Optional[str] = None,
    lowercase: bool = True,
    replace: Optional[dict] = None,
    append: Optional[str | List[str]] = None,
    infer: Optional[dict] = None,
    donor_colname: Optional[str] = None,
    transfer: Optional[str | list[str]] = None,
) -> pl.LazyFrame:
    """
    Indicator type is the survey question that was asked.
    An override indicator type can also be given to fill in all rows.
    """
    df = (
        df.pipe(_replace_column_name, "indicator_type", colname, override)
        .pipe(
            _replace_column_values, "indicator_type", lowercase, replace, append, infer
        )
        .pipe(_borrow_column_values, "indicator_type", donor_colname, transfer)
    )

    return df


def clean_indicator(
    df: pl.LazyFrame,
    colname: str | None,
    override: Optional[str] = None,
    lowercase: bool = False,
    replace: Optional[dict] = None,
    append: Optional[str | List[str]] = None,
    infer: Optional[dict] = None,
    donor_colname: Optional[str] = None,
    transfer: Optional[str | list[str]] = None,
) -> pl.LazyFrame:
    """
    Indicator is the specific answer to the survey question.
    Synonyms is a list of (indicator_type, indicator) tuples that are identical.
    Only the synonym with the most rows will be kept, using the verbiage of the first synonym.
    E.g. ("4-level vaccination and intent", "Received a vaccination") and ("up-to-date", "Yes")
    are synonymous, so the former should be kept and the latter discarded.
    An override indicator can also be given to fill in all rows.
    """
    df = (
        df.pipe(_replace_column_name, "indicator", colname, override)
        .pipe(_replace_column_values, "indicator", lowercase, replace, append, infer)
        .pipe(_borrow_column_values, "indicator", donor_colname, transfer)
    )
    # if synonyms is not None:
    #     sub_dfs = pl.collect_all(
    #         [
    #             df.filter(
    #                 (pl.col("indicator_type") == pair[0])
    #                 & (pl.col("indicator") == pair[1])
    #             ).drop(["indicator_type", "indicator"])
    #             for pair in synonyms
    #         ]
    #     )
    #     ref_idx = max(range(len(sub_dfs)), key=lambda idx: sub_dfs[idx].height)
    #     ref_df = sub_dfs[ref_idx]
    #     for sub_df in sub_dfs:
    #         extra_rows = sub_df.join(ref_df, on=ref_df.columns, how="anti")
    #         if extra_rows.height > 0:
    #             raise RuntimeError("Indicator pairs are not synonymous", extra_rows)
    #     ref_pair = synonyms[ref_idx]
    #     pref_pair = synonyms[0]
    #     synonyms.pop(ref_idx)
    #     df = df.filter(
    #         [
    #             (pl.col("indicator_type") != pair[0]) | (pl.col("indicator") != pair[1])
    #             for pair in synonyms
    #         ]
    #     )
    #     df = df.with_columns(
    #         indicator_type=pl.when(
    #             (pl.col("indicator_type") == ref_pair[0])
    #             & (pl.col("indicator") == ref_pair[1])
    #         )
    #         .then(pl.lit(pref_pair[0]))
    #         .otherwise(pl.col("indicator_type")),
    #         indicator=pl.when(
    #             (pl.col("indicator_type") == ref_pair[0])
    #             & (pl.col("indicator") == ref_pair[1])
    #         )
    #         .then(pl.lit(pref_pair[1]))
    #         .otherwise(pl.col("indicator")),
    #     )

    return df


def clean_vaccine(
    df: pl.LazyFrame,
    colname: str | None,
    override: Optional[str] = None,
    lowercase: bool = True,
    replace: Optional[dict] = None,
    append: Optional[str | List[str]] = None,
    infer: Optional[dict] = None,
    donor_colname: Optional[str] = None,
    transfer: Optional[str | list[str]] = None,
) -> pl.LazyFrame:
    """
    Vaccine is the target pathogen plus any formulation information.
    If an 'infer' dictionary is given along with a column, the vaccine
    is inferred from that column: in the dictionary, keys = phrases to
    look for, and values = the vaccines those phrases indicate.
    If there is no column with vaccine information, provide it as 'override'.
    Move extraneous information about eligibilty, etc. to the 'domain'
    column as necessary by specifying the extraneous phrases.

    """
    df = (
        df.pipe(_replace_column_name, "vaccine", colname, override)
        .pipe(_replace_column_values, "vaccine", lowercase, replace, append, infer)
        .pipe(_borrow_column_values, "vaccine", donor_colname, transfer)
    )
    # if infer is not None:
    #     expr = pl.lit("Unrecognized vaccine")
    #     for phrase, vax in infer.items():
    #         expr = (
    #             pl.when(pl.col("vaccine").str.contains(phrase))
    #             .then(pl.lit(vax))
    #             .otherwise(expr)
    #         )
    #     df = df.with_columns(vaccine=expr)
    # if domain_phrases is not None:
    #     for phrase in domain_phrases:
    #         df = df.with_columns(
    #             domain=pl.when(pl.col("vaccine").str.contains(phrase))
    #             .then(pl.col("domain") + phrase)
    #             .otherwise(pl.col("domain")),
    #             vaccine=pl.col("vaccine").str.replace("phrase", ""),
    #         )
    # df = df.with_columns(pl.col("vaccine").str.strip_chars())

    return df


def clean_time_type(
    df: pl.LazyFrame,
    colname: str | None,
    override: Optional[str] = None,
    lowercase: bool = True,
    replace: Optional[dict] = None,
    append: Optional[str | List[str]] = None,
    infer: Optional[dict] = None,
    donor_colname: Optional[str] = None,
    transfer: Optional[str | list[str]] = None,
) -> pl.LazyFrame:
    """
    Time type is the interval between report dates, e.g. 'month' or 'week'.
    If there is no column with this information, provide it as 'override'.
    """
    if replace is None:
        replace = {"ly": ""}
    df = (
        df.pipe(_replace_column_name, "time_type", colname, override)
        .pipe(_replace_column_values, "time_type", lowercase, replace, append, infer)
        .pipe(_borrow_column_values, "time_type", donor_colname, transfer)
    )

    return df


def clean_time_start_end(
    df: pl.LazyFrame,
    column: str | List[str],
    col_format: str = "end",
    time_format: str = "%Y-%m-%dT%H:%M:%S%.f",
) -> pl.LazyFrame:
    """
    Time start is the date on which phone surveys began for the reported estimate.
    Time end is the date on which phone surveys ended for the reported estimate.
    A list of columns may be given if month-day is in one column and year in another.
    Column format is "start", "end", or "both" depending on which times are given.
    Time format is the format of the date string once it is constructed.
    """
    if not isinstance(column, list):
        column = [column]
    if len(column) > 1:
        df = df.with_columns(pl.col(column[1]).str.strip_chars().str.slice(0, 4))
    if col_format == "end":
        if len(column) == 1:
            df = df.with_columns(
                time_end=pl.col(column[0])
                .str.strptime(pl.Date, time_format)
                .dt.truncate("1d")
            )
        elif len(column) > 1:
            df = df.with_columns(
                time_end=pl.concat_str(
                    [column[0], column[1]], separator="-"
                ).str.strptime(pl.Date, time_format)
            )
        df = df.with_columns(
            time_start=pl.when(pl.col("time_type") == "week")
            .then(pl.col("time_end").dt.offset_by("-6d"))
            .when(pl.col("time_type") == "month")
            .then(pl.col("time_end").dt.offset_by("-1mo"))
        )
        if (
            df.filter(~pl.col("time_type").is_in(["week", "month"])).collect().shape[0]
            > 0
        ):
            odd_time_type = (
                df.filter(~pl.col("time_type").is_in(["week", "month"]))
                .collect()
                .select("time_type")
                .unique()
            )
            raise RuntimeError("Time type not recognized:", odd_time_type)
    elif col_format == "both":
        df = df.with_columns(
            time_start=pl.col(column[0]).str.extract(r"^(.*?)-").str.strip_chars(),
            time_end=pl.col(column[0]).str.extract(r"^(.*?)-").str.strip_chars(),
        )
        if len(column) > 1:
            df = df.with_columns(
                time_start=(pl.col("time_start") + " " + pl.col(column[1])),
                time_end=(pl.col("time_end") + " " + pl.col(column[1])),
            )
        df = df.with_columns(
            pl.col("time_start").str.strptime(pl.Date, time_format).dt.truncate("1d"),
            pl.col("time_end").str.strptime(pl.Date, time_format).dt.truncate("1d"),
        )
    else:
        raise RuntimeError(f"Column format {col_format} is not recognized.")

    return df


def clean_estimate(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """
    Estimate is the percentage of respondents represented by a row.
    """
    df = df.rename({column: "estimate"})
    bad_estimates = df.filter(pl.col("estimate").str.contains(r"[a-zA-Z]")).collect()
    if bad_estimates.shape[0] > 0:
        warnings.warn(
            "Some rows contain non-numeric estimates. These rows will be dropped."
        )
        print(bad_estimates)
    df = df.filter(~pl.col("estimate").str.contains(r"[a-zA-Z]"))
    df = df.with_columns(
        estimate=((pl.col("estimate").cast(pl.Float64)) / 100.0).clip(
            lower_bound=0.0, upper_bound=1.0
        )
    )

    return df


def clean_lci_uci(
    df: pl.LazyFrame,
    column: str,
    col_format: str = "half",
    separator: str = "-",
) -> pl.LazyFrame:
    """
    LCI and UCI are the lower & upper 95% confidence intervals on the estimate.
    Column format is "half" or "full" depending on whether the CI half-width or
    full range is given. In the latter case, specify the separating character(s).
    """
    bad_cis = df.filter(pl.col(column).str.contains("NA")).collect()
    if bad_cis.shape[0] > 0:
        warnings.warn("Some rows contain NA CIs. These rows will be dropped.")
        print(bad_cis)
    df = df.filter(~pl.col(column).str.contains("NA"))
    if col_format == "half":
        df = (
            df.with_columns(pl.col(column).cast(pl.Float64) / 100.0)
            .with_columns(
                lci=(pl.col("estimate") - pl.col(column)).clip(lower_bound=0.0),
                uci=(pl.col("estimate") + pl.col(column)).clip(upper_bound=1.0),
            )
            .drop(column)
        )
    elif col_format == "full":
        df = (
            df.with_columns(
                pl.col(column)
                .str.replace(separator, "-")
                .str.replace(r" â€¡$", "")
                .str.replace(r" ‡$", "")
            )
            .with_columns(
                lci=(
                    pl.col(column)
                    .str.extract(r"^(.*?)-")
                    .str.strip_chars()
                    .cast(pl.Float64)
                )
                / 100.0,
                uci=(
                    pl.col(column)
                    .str.extract(r"-(.*)", 1)
                    .str.strip_chars()
                    .cast(pl.Float64)
                )
                / 100.0,
            )
            .drop(column)
        )
    else:
        raise RuntimeError(f"Column format {col_format} is not recognized.")

    return df


def clean_sample_size(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
    df = df.rename({column: "sample_size"})
    df = df.with_columns(pl.col("sample_size").cast(pl.UInt32))

    return df


def remove_duplicates(
    df: pl.LazyFrame,
    tolerance: float = 0.001,
    synonym_columns: Optional[Tuple] = None,
    synonyms: Optional[List[Tuple]] = None,
) -> pl.LazyFrame:
    """
    Rows are duplicates if they are within some tolerance for value columns
    (estimate, lci, & uci) and identical for group columns (all others).
    To find duplicates, group by group columns and get mean of value columns.
    Then verify that the difference between the raw and mean values < tolerance.
    If duplicate rows are found, average their values together.
    If duplicate groups have clashing values, raise an error.
    """
    if synonym_columns is not None:
        if synonyms is None:
            raise RuntimeError(
                "If 'synonym columns' are provided, 'synonyms' must be too."
            )
        assert (len(s) == len(synonym_columns) for s in synonyms)
        sub_dfs = pl.collect_all(
            [
                df.filter(
                    (pl.col(col) == syn) for col, syn in zip(synonym_columns, synonym)
                )
                for synonym in synonyms
            ]
        )
        ref_idx = max(range(len(sub_dfs)), key=lambda idx: sub_dfs[idx].height)
        ref_df = sub_dfs[ref_idx].drop(synonym_columns)
        for sub_df in sub_dfs:
            extra_rows = sub_df.drop(synonym_columns).join(
                ref_df, on=ref_df.columns, how="anti"
            )
            if extra_rows.height > 0:
                raise RuntimeError(
                    "Declared synonyms are not really synonyms.", extra_rows
                )
        ref_df = ref_df.with_columns(
            [pl.lit(syn).alias(col) for col, syn in zip(synonym_columns, synonyms[0])]
        )
        df = pl.concat(
            [
                df.collect().join(
                    pl.concat(sub_dfs),
                    on=list(set(df.columns) - set(synonym_columns)),
                    how="anti",
                ),
                ref_df,
            ]
        ).lazy()

    value_columns = {"estimate", "lci", "uci"}
    group_columns = data_schema.keys() - value_columns

    bad_groups = (
        df.group_by(group_columns)
        .agg(pl.col(value_columns).pipe(_mean_max_diff, tolerance=tolerance))
        .filter(pl.all_horizontal(value_columns).not_())
        .collect()
    )

    if bad_groups.shape[0] > 0:
        raise RuntimeError("Some identical groups have clashing values:", bad_groups)

    return df.group_by(group_columns).agg(pl.col(value_columns).mean())


def _mean_max_diff(x: pl.Expr, tolerance: float) -> pl.Expr:
    return (x - x.mean()).abs().max() < tolerance


def _replace_column_name(
    df: pl.LazyFrame,
    new_colname: str,
    old_colname: Optional[str] = None,
    override: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Create a new column in a data frame, either by:
    - renaming an old column and keeping the original values, or
    - creating a new column and filling in a repeated value
    """
    if old_colname is not None:
        df = df.rename({old_colname: new_colname})
        if override is not None:
            raise RuntimeError(
                "Exactly one of 'old_colname' or 'override_value' must be given."
            )
    else:
        if override is None:
            raise RuntimeError(
                "Exactly one of 'old_colname' or 'override_value' must be given."
            )
        df = df.with_columns(pl.lit(override).alias(new_colname))

    return df


def _replace_column_values(
    df: pl.LazyFrame,
    colname: str,
    lowercase: bool = True,
    replace: Optional[dict] = None,
    append: Optional[str | List[str]] = None,
    infer: Optional[dict] = None,
) -> pl.LazyFrame:
    """
    Replace the values in a column by:
    - removing leading/trailing whitespace
    - setting to lowercase if desired, and
    - replacing certain strings with others
    - appending strings if they are missing
    - inferring entirely new values from the presence of certain strings
    """
    new_values = pl.col(colname).str.strip_chars()
    if lowercase:
        new_values = new_values.str.to_lowercase()
    if replace is not None:
        new_values = new_values.str.replace_many(replace)
    if append is not None:
        if not isinstance(append, list):
            append = [append]
        for phrase in append:
            new_values = (
                pl.when(new_values.str.contains(phrase))
                .then(new_values)
                .otherwise(new_values + " & " + " & ".join(phrase))
            )
    if infer is not None:
        for old_phrase, new_phrase in infer.items():
            new_values = (
                pl.when(new_values.str.contains(old_phrase))
                .then(pl.lit(new_phrase))
                .otherwise(new_values)
            )

    df = df.with_columns(new_values)

    return df


def _borrow_column_values(
    df: pl.LazyFrame,
    recip_colname: str,
    donor_colname: Optional[str] = None,
    transfer: Optional[str | List[str]] = None,
) -> pl.LazyFrame:
    """
    Augment the values in a recipient column with the values in a donor column by
    - Transferring the whole donor column to the recipient, when the two are not already identical.
    - Transferring phrases from the donor column to the recipient, when those phrases are present.
    """
    if donor_colname is None:
        return df

    if transfer is None:
        new_values = (
            pl.when(pl.col(recip_colname) != pl.col(donor_colname))
            .then(pl.concat_str([recip_colname, donor_colname], separator=" & "))
            .otherwise(pl.col(recip_colname))
        )
    else:
        if not isinstance(transfer, list):
            transfer = [transfer]
        for phrase in transfer:
            new_values = (
                pl.when(pl.col(donor_colname).str.contains(phrase))
                .then(pl.col(recip_colname) + phrase)
                .otherwise(pl.col(recip_colname))
            )

    df = df.with_columns(new_values.alias(recip_colname))

    return df


def enforce_schema(df: pl.LazyFrame, schema: pl.Schema = data_schema) -> pl.LazyFrame:
    """
    Enforce that the standardized schema is followed. Remove extra columns.
    """
    current_columns = df.collect_schema().names()
    needed_columns = schema.names()
    missing_columns = set(needed_columns) - set(current_columns)
    extra_columns = set(current_columns) - set(needed_columns)
    if missing_columns != set():
        raise RuntimeError("Missing columns:", missing_columns)
    if extra_columns != set():
        warnings.warn(f"Dropped columns: {extra_columns}")
    return df.select(needed_columns)


def ensure_eager(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, pl.LazyFrame):
        return df.collect()
    else:
        raise RuntimeError(f"Cannot collect object of type {type(df)}")


def duplicated_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Return duplicated rows of an (eager) data frame

    Args:
        df (pl.DataFrame): input df

    Returns:
        pl.DataFrame: duplicated rows only
    """
    return df.filter(df.is_duplicated())


def rows_with_any_null(df: pl.DataFrame) -> pl.DataFrame:
    """Filter a data frame for rows with any null value

    Args:
        df (pl.LazyFrame): data frame

    Returns:
        pl.LazyFrame: rows with any null value
    """
    return df.filter(pl.any_horizontal(pl.all().is_null()))
