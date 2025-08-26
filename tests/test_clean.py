import polars as pl
import polars.testing
import pytest

from nisapi.clean import Validate
from nisapi.clean.helpers import (
    _replace_column_name,
    _replace_column_values,
    _borrow_column_values,
    clean_lci_uci,
    clean_sample_size,
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
            "estimate": [1.0, 10.0, 1.0, 10.0, 2.0, 2.0, 6.0, 6.0],
            "_ci_95": [0.1, 0.1, 1.0, 10.2, 0.2, 0.2, 0.6, 0.6],
            "ss": [100, 100, 1000, 1000, 100, 100, 100, 100],
        }
    )
    return df


def test_drop_bad_rows(mock_df):
    result = drop_bad_rows(
        mock_df.lazy(), colname="supp_flag", bad_columns=["_ci_95", "ss"]
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
            "estimate": [1.0, 10.0, 1.0, 10.0, 2.0, 2.0, 6.0],
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
            "estimate": [1.0, 10.0, 1.0, 10.0, 2.0, 2.0, 6.0, 6.0],
            "_ci_95": [0.1, 0.1, 1.0, 10.2, 0.2, 0.2, 0.6, 0.6],
            "ss": [100, 100, 1000, 1000, 100, 100, 100, 100],
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
            "estimate": [1.0, 10.0, 1.0, 10.0, 2.0, 2.0, 6.0, 6.0],
            "_ci_95": [0.1, 0.1, 1.0, 10.2, 0.2, 0.2, 0.6, 0.6],
            "ss": [100, 100, 1000, 1000, 100, 100, 100, 100],
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
            "estimate": [1.0, 10.0, 1.0, 10.0, 2.0, 2.0, 6.0, 6.0],
            "_ci_95": [0.1, 0.1, 1.0, 10.2, 0.2, 0.2, 0.6, 0.6],
            "ss": [100, 100, 1000, 1000, 100, 100, 100, 100],
            "time_start": ["2025-07-26"] * 8,
            "time_end": ["2025-08-26"] * 8,
        }
    ).with_columns(
        time_end=pl.col("time_end").str.to_date(),
        time_start=pl.col("time_start").str.to_date(),
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
