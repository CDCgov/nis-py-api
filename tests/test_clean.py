import polars as pl
import polars.testing
from nisapi.clean.helpers import remove_near_duplicates, _mean_max_diff


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


def test_remove_near_duplicates_one_value():
    input_df = pl.DataFrame(
        {
            "group": [1, 1, 2, 2],
            "value1": [0.0, 0.1, 1.0, 1.1],
            "value2": [2.0, 2.1, 3.0, 3.1],
        }
    )

    current_df = input_df.pipe(
        remove_near_duplicates,
        value_columns=["value1", "value2"],
        group_columns=["group"],
        tolerance=0.1,
        n_fold_duplication=2,
    )
    expected_df = pl.DataFrame(
        {"group": [1, 2], "value1": [0.05, 1.05], "value2": [2.05, 3.05]}
    )
    polars.testing.assert_frame_equal(
        current_df, expected_df, check_column_order=False, check_row_order=False
    )


def test_remove_near_duplicates_multiple_values():
    input_df = pl.DataFrame(
        {
            "group": [1, 1, 2, 2],
            "value1": [0.12, 0.1, 1.12, 1.1],
            "value2": [2.0, 2.01, 3.0, 3.01],
        }
    )

    current_df = input_df.pipe(
        remove_near_duplicates,
        value_columns=["value1", "value2"],
        group_columns=["group"],
        tolerance=0.1,
        n_fold_duplication=2,
    )
    expected_df = pl.DataFrame(
        {"group": [1, 2], "value1": [0.11, 1.11], "value2": [2.005, 3.005]}
    )
    polars.testing.assert_frame_equal(
        current_df, expected_df, check_column_order=False, check_row_order=False
    )
