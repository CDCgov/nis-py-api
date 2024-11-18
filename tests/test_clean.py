import polars as pl
import polars.testing
from nisapi.clean import remove_near_duplicates


def test_remove_near_duplicates_first():
    input_df = pl.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "group": [1, 1, 2, 2],
            "value1": [0.0, 0.01, 1.0, 1.01],
            "value2": [2.0, 2.01, 3.0, 3.01],
        }
    )

    current_df = input_df.pipe(
        remove_near_duplicates,
        filter_expr=pl.col("value1") == pl.col("value1").first(),
        value_columns=["value1", "value2"],
        group_columns=["group"],
        tolerance=1e-1,
        n_fold_duplication=2,
    )
    expected_df = input_df.filter(pl.col("row_id").is_in([1, 3]))
    polars.testing.assert_frame_equal(current_df, expected_df, check_column_order=False)


def test_remove_near_duplicates_round():
    input_df = pl.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "group": [1, 1, 2, 2],
            "value1": [0.123, 0.1, 1.123, 1.1],
            "value2": [2.0, 2.01, 3.0, 3.01],
        }
    )

    current_df = input_df.pipe(
        remove_near_duplicates,
        filter_expr=pl.col("value1") == pl.col("value1").round(1),
        value_columns=["value1", "value2"],
        group_columns=["group"],
        tolerance=1e-1,
        n_fold_duplication=2,
    )
    expected_df = input_df.filter(pl.col("row_id").is_in([2, 4]))
    polars.testing.assert_frame_equal(current_df, expected_df, check_column_order=False)
