import polars as pl
import polars.testing
import pytest

from nisapi.clean import Validate
from nisapi.clean.helpers import (
    _replace_column_name,
    _replace_column_values,
    _borrow_column_values,
    clean_estimate,
    clean_lci_uci,
    clean_time_start_end,
    drop_bad_rows,
    _mean_max_diff,
    remove_duplicates,
    rows_with_any_null,
)


@pytest.fixture
def mock_df():
    df = pl.DataFrame(
        {
            "supp_flag": ["0", "0", "0", "0", "0", "0", "0", "1"],
            "text_col1": ["area", "area", "area", "area", "age", "age", "age", "age"],
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45", " 18 - 45"],
            "time_type": ["month"] * 8,
            "time": ["2025-08-26"] * 8,
            "time_range": ["July 26 2025 - August 26 2025"] * 8,
            "month_day": ["July 26 - August 26"] * 8,
            "year": ["2025"] * 8,
            "estimate": ["1.0", "10.0", "1.0", "10.0", "2.0", "2.0", "6.0", "six"],
            "_ci_95": ["0.1", "1.0", "0.1", "10.2", "0.2", "0.2", "0.6", "0.6"],
            "ci": ["0.0 to 100.2"] * 8,
        }
    )
    return df


def test_drop_bad_rows(mock_df):
    result = drop_bad_rows(
        mock_df.lazy(), colname="supp_flag", bad_columns=["_ci_95", "ci"]
    ).collect()

    expected = pl.DataFrame(
        {
            "text_col1": ["area", "area", "area", "area", "age", "age", "age"],
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45"],
            "time_type": ["month"] * 7,
            "time": ["2025-08-26"] * 7,
            "time_range": ["July 26 2025 - August 26 2025"] * 7,
            "month_day": ["July 26 - August 26"] * 7,
            "year": ["2025"] * 7,
            "estimate": ["1.0", "10.0", "1.0", "10.0", "2.0", "2.0", "6.0"],
        }
    )

    polars.testing.assert_frame_equal(result, expected, check_row_order=False)


def test_clean_time_start_end_endcol(mock_df):
    result = clean_time_start_end(mock_df.lazy(), "time", "end", "%Y-%m-%d").collect()

    expected = pl.DataFrame(
        {
            "supp_flag": ["0", "0", "0", "0", "0", "0", "0", "1"],
            "text_col1": ["area", "area", "area", "area", "age", "age", "age", "age"],
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45", " 18 - 45"],
            "time_type": ["month"] * 8,
            "time": ["2025-08-26"] * 8,
            "time_range": ["July 26 2025 - August 26 2025"] * 8,
            "month_day": ["July 26 - August 26"] * 8,
            "year": ["2025"] * 8,
            "estimate": ["1.0", "10.0", "1.0", "10.0", "2.0", "2.0", "6.0", "six"],
            "_ci_95": ["0.1", "1.0", "0.1", "10.2", "0.2", "0.2", "0.6", "0.6"],
            "ci": ["0.0 to 100.2"] * 8,
            "time_end": ["2025-08-26"] * 8,
            "time_start": ["2025-07-26"] * 8,
        }
    ).with_columns(
        time_end=pl.col("time_end").str.to_date(),
        time_start=pl.col("time_start").str.to_date(),
    )

    polars.testing.assert_frame_equal(result, expected, check_row_order=False)


def test_clean_time_start_end_bothcol(mock_df):
    result = clean_time_start_end(
        mock_df.lazy(), "time_range", "both", "%B %d %Y"
    ).collect()

    expected = pl.DataFrame(
        {
            "supp_flag": ["0", "0", "0", "0", "0", "0", "0", "1"],
            "text_col1": ["area", "area", "area", "area", "age", "age", "age", "age"],
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45", " 18 - 45"],
            "time_type": ["month"] * 8,
            "time": ["2025-08-26"] * 8,
            "time_range": ["July 26 2025 - August 26 2025"] * 8,
            "month_day": ["July 26 - August 26"] * 8,
            "year": ["2025"] * 8,
            "estimate": ["1.0", "10.0", "1.0", "10.0", "2.0", "2.0", "6.0", "six"],
            "_ci_95": ["0.1", "1.0", "0.1", "10.2", "0.2", "0.2", "0.6", "0.6"],
            "ci": ["0.0 to 100.2"] * 8,
            "time_start": ["2025-07-26"] * 8,
            "time_end": ["2025-08-26"] * 8,
        }
    ).with_columns(
        time_end=pl.col("time_end").str.to_date(),
        time_start=pl.col("time_start").str.to_date(),
    )

    polars.testing.assert_frame_equal(result, expected, check_row_order=False)


def test_clean_time_start_end_twocol(mock_df):
    result = clean_time_start_end(
        mock_df.lazy(), ["month_day", "year"], "both", "%B %d %Y"
    ).collect()

    expected = pl.DataFrame(
        {
            "supp_flag": ["0", "0", "0", "0", "0", "0", "0", "1"],
            "text_col1": ["area", "area", "area", "area", "age", "age", "age", "age"],
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45", " 18 - 45"],
            "time_type": ["month"] * 8,
            "time": ["2025-08-26"] * 8,
            "time_range": ["July 26 2025 - August 26 2025"] * 8,
            "month_day": ["July 26 - August 26"] * 8,
            "year": ["2025"] * 8,
            "estimate": ["1.0", "10.0", "1.0", "10.0", "2.0", "2.0", "6.0", "six"],
            "_ci_95": ["0.1", "1.0", "0.1", "10.2", "0.2", "0.2", "0.6", "0.6"],
            "ci": ["0.0 to 100.2"] * 8,
            "time_start": ["2025-07-26"] * 8,
            "time_end": ["2025-08-26"] * 8,
        }
    ).with_columns(
        time_end=pl.col("time_end").str.to_date(),
        time_start=pl.col("time_start").str.to_date(),
    )

    polars.testing.assert_frame_equal(result, expected, check_row_order=False)


def test_clean_estimate(mock_df):
    result = clean_estimate(mock_df.lazy(), "estimate").collect()

    expected = pl.DataFrame(
        {
            "supp_flag": ["0", "0", "0", "0", "0", "0", "0"],
            "text_col1": ["area", "area", "area", "area", "age", "age", "age"],
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45"],
            "time_type": ["month"] * 7,
            "time": ["2025-08-26"] * 7,
            "time_range": ["July 26 2025 - August 26 2025"] * 7,
            "month_day": ["July 26 - August 26"] * 7,
            "year": ["2025"] * 7,
            "estimate": [0.01, 0.1, 0.01, 0.1, 0.02, 0.02, 0.06],
            "_ci_95": ["0.1", "1.0", "0.1", "10.2", "0.2", "0.2", "0.6"],
            "ci": ["0.0 to 100.2"] * 7,
        }
    )

    polars.testing.assert_frame_equal(result, expected, check_row_order=False)


def test_clean_lci_uci_half(mock_df):
    result = clean_estimate(mock_df.lazy(), "estimate")
    result = clean_lci_uci(result, "_ci_95", "half").collect()

    expected = pl.DataFrame(
        {
            "supp_flag": ["0", "0", "0", "0", "0", "0", "0"],
            "text_col1": ["area", "area", "area", "area", "age", "age", "age"],
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45"],
            "time_type": ["month"] * 7,
            "time": ["2025-08-26"] * 7,
            "time_range": ["July 26 2025 - August 26 2025"] * 7,
            "month_day": ["July 26 - August 26"] * 7,
            "year": ["2025"] * 7,
            "estimate": [0.01, 0.1, 0.01, 0.1, 0.02, 0.02, 0.06],
            "ci": ["0.0 to 100.2"] * 7,
            "lci": [0.009, 0.09, 0.009, 0.0, 0.018, 0.018, 0.054],
            "uci": [0.011, 0.11, 0.011, 0.202, 0.022, 0.022, 0.066],
        }
    )

    polars.testing.assert_frame_equal(result, expected, check_row_order=False)


def test_clean_lci_uci_full(mock_df):
    result = clean_estimate(mock_df.lazy(), "estimate")
    result = clean_lci_uci(result, "ci", "full", "to").collect()

    expected = pl.DataFrame(
        {
            "supp_flag": ["0", "0", "0", "0", "0", "0", "0"],
            "text_col1": ["area", "area", "area", "area", "age", "age", "age"],
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45"],
            "time_type": ["month"] * 7,
            "time": ["2025-08-26"] * 7,
            "time_range": ["July 26 2025 - August 26 2025"] * 7,
            "month_day": ["July 26 - August 26"] * 7,
            "year": ["2025"] * 7,
            "estimate": [0.01, 0.1, 0.01, 0.1, 0.02, 0.02, 0.06],
            "_ci_95": ["0.1", "1.0", "0.1", "10.2", "0.2", "0.2", "0.6"],
            "lci": [0.0] * 7,
            "uci": [1.0] * 7,
        }
    )

    polars.testing.assert_frame_equal(result, expected, check_row_order=False)


def test_replace_column_name_rename(mock_df):
    result = _replace_column_name(mock_df, "new_name", "text_col1")

    expected = pl.DataFrame(
        {
            "supp_flag": ["0", "0", "0", "0", "0", "0", "0", "1"],
            "new_name": ["area", "area", "area", "area", "age", "age", "age", "age"],
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45", " 18 - 45"],
            "time_type": ["month"] * 8,
            "time": ["2025-08-26"] * 8,
            "time_range": ["July 26 2025 - August 26 2025"] * 8,
            "month_day": ["July 26 - August 26"] * 8,
            "year": ["2025"] * 8,
            "estimate": ["1.0", "10.0", "1.0", "10.0", "2.0", "2.0", "6.0", "six"],
            "_ci_95": ["0.1", "1.0", "0.1", "10.2", "0.2", "0.2", "0.6", "0.6"],
            "ci": ["0.0 to 100.2"] * 8,
        }
    )

    polars.testing.assert_frame_equal(result, expected, check_row_order=False)


def test_replace_column_name_override(mock_df):
    result = _replace_column_name(mock_df, "text_col1", override="some_category_type")

    expected = pl.DataFrame(
        {
            "supp_flag": ["0", "0", "0", "0", "0", "0", "0", "1"],
            "text_col1": ["some_category_type"] * 8,
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45", " 18 - 45"],
            "time_type": ["month"] * 8,
            "time": ["2025-08-26"] * 8,
            "time_range": ["July 26 2025 - August 26 2025"] * 8,
            "month_day": ["July 26 - August 26"] * 8,
            "year": ["2025"] * 8,
            "estimate": ["1.0", "10.0", "1.0", "10.0", "2.0", "2.0", "6.0", "six"],
            "_ci_95": ["0.1", "1.0", "0.1", "10.2", "0.2", "0.2", "0.6", "0.6"],
            "ci": ["0.0 to 100.2"] * 8,
        }
    )

    polars.testing.assert_frame_equal(result, expected, check_row_order=False)


def test_replace_column_values_replace(mock_df):
    result = _replace_column_values(
        mock_df, "text_col1", replace={"area": "Area", "age": "Age"}
    )

    expected = pl.DataFrame(
        {
            "supp_flag": ["0", "0", "0", "0", "0", "0", "0", "1"],
            "text_col1": ["Area", "Area", "Area", "Area", "Age", "Age", "Age", "Age"],
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45", " 18 - 45"],
            "time_type": ["month"] * 8,
            "time": ["2025-08-26"] * 8,
            "time_range": ["July 26 2025 - August 26 2025"] * 8,
            "month_day": ["July 26 - August 26"] * 8,
            "year": ["2025"] * 8,
            "estimate": ["1.0", "10.0", "1.0", "10.0", "2.0", "2.0", "6.0", "six"],
            "_ci_95": ["0.1", "1.0", "0.1", "10.2", "0.2", "0.2", "0.6", "0.6"],
            "ci": ["0.0 to 100.2"] * 8,
        }
    )

    polars.testing.assert_frame_equal(result, expected, check_row_order=False)


def test_replace_column_values_append(mock_df):
    result = _replace_column_values(mock_df, "text_col1", append="age")

    expected = pl.DataFrame(
        {
            "supp_flag": ["0", "0", "0", "0", "0", "0", "0", "1"],
            "text_col1": [
                "area & age",
                "area & age",
                "area & age",
                "area & age",
                "age",
                "age",
                "age",
                "age",
            ],
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45", " 18 - 45"],
            "time_type": ["month"] * 8,
            "time": ["2025-08-26"] * 8,
            "time_range": ["July 26 2025 - August 26 2025"] * 8,
            "month_day": ["July 26 - August 26"] * 8,
            "year": ["2025"] * 8,
            "estimate": ["1.0", "10.0", "1.0", "10.0", "2.0", "2.0", "6.0", "six"],
            "_ci_95": ["0.1", "1.0", "0.1", "10.2", "0.2", "0.2", "0.6", "0.6"],
            "ci": ["0.0 to 100.2"] * 8,
        }
    )

    polars.testing.assert_frame_equal(result, expected, check_row_order=False)


def test_replace_column_values_infer(mock_df):
    result = _replace_column_values(mock_df, "text_col1", infer={"area": "region"})

    expected = pl.DataFrame(
        {
            "supp_flag": ["0", "0", "0", "0", "0", "0", "0", "1"],
            "text_col1": [
                "region",
                "region",
                "region",
                "region",
                "age",
                "age",
                "age",
                "age",
            ],
            "text_col2": ["PA", "US", "PA", "US", "18+", " 18+ ", "18- 45", " 18 - 45"],
            "time_type": ["month"] * 8,
            "time": ["2025-08-26"] * 8,
            "time_range": ["July 26 2025 - August 26 2025"] * 8,
            "month_day": ["July 26 - August 26"] * 8,
            "year": ["2025"] * 8,
            "estimate": ["1.0", "10.0", "1.0", "10.0", "2.0", "2.0", "6.0", "six"],
            "_ci_95": ["0.1", "1.0", "0.1", "10.2", "0.2", "0.2", "0.6", "0.6"],
            "ci": ["0.0 to 100.2"] * 8,
        }
    )

    polars.testing.assert_frame_equal(result, expected, check_row_order=False)


def test_mean_max_diff():
    input_df = pl.DataFrame(
        {
            "group": [1, 1, 2, 2],
            "value": [0.0, 2.0, 0.0, 20.0],
        }
    )

    # all should pass for large tolerance
    current = input_df.group_by("group").agg(
        pl.col("value").pipe(_mean_max_diff, tolerance=100.0)
    )
    expected = pl.DataFrame({"group": [1, 2], "value": [True, True]})
    polars.testing.assert_frame_equal(current, expected, check_row_order=False)

    # should fail for small tolerance
    current = input_df.group_by("group").agg(
        pl.col("value").pipe(_mean_max_diff, tolerance=2.0)
    )
    expected = pl.DataFrame({"group": [1, 2], "value": [True, False]})
    polars.testing.assert_frame_equal(current, expected, check_row_order=False)


def test_remove_duplicates():
    input_df = pl.DataFrame(
        {
            "estimate": [1.0, 1.1, 2.0, 2.1],
            "lci": [0.0, 0.1, 1.0, 1.1],
            "uci": [2.0, 2.1, 3.0, 3.1],
        }
    ).with_columns(
        [
            pl.Series(name, ["A", "A", "B", "B"])
            for name in [
                "geography_type",
                "geography",
                "domain_type",
                "domain",
                "indicator_type",
                "indicator",
                "vaccine",
                "time_type",
                "time_start",
                "time_end",
                "sample_size",
            ]
        ]
    )

    current_df = (
        input_df.lazy()
        .pipe(
            remove_duplicates,
            tolerance=0.1,
        )
        .collect()
    )
    expected_df = pl.DataFrame(
        {"estimate": [1.05, 2.05], "lci": [0.05, 1.05], "uci": [2.05, 3.05]}
    ).with_columns(
        [
            pl.Series(name, ["A", "B"])
            for name in [
                "geography_type",
                "geography",
                "domain_type",
                "domain",
                "indicator_type",
                "indicator",
                "vaccine",
                "time_type",
                "time_start",
                "time_end",
                "sample_size",
            ]
        ]
    )
    polars.testing.assert_frame_equal(
        current_df, expected_df, check_column_order=False, check_row_order=False
    )


def test_validate_age_groups():
    assert (
        pl.DataFrame({"age_group": ["18-49 years", "50-64 years", "65+ years"]})
        .select(Validate.is_valid_age_group(pl.col("age_group")).all())
        .item()
    )

    assert (
        not pl.DataFrame(
            {
                "age_group": [
                    # en dash should fail
                    "18–49 years"
                    # missing "years" should fail
                    "18-49",
                    # fail if there are spaces
                    "18 - 49 years",
                ]
            }
        )
        .select(Validate.is_valid_age_group(pl.col("age_group")).any())
        .item()
    )


def test_row_has_null():
    df = pl.DataFrame(
        {"id": [1, 2, 3, 4], "x": [None, None, 3, 4], "y": [1, None, None, 4]}
    )
    current = df.pipe(rows_with_any_null)
    expected = df.filter(pl.col("id").is_in([1, 2, 3]))

    polars.testing.assert_frame_equal(current, expected)
