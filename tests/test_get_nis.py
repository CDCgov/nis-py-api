import nisapi
import polars as pl
from polars.testing import assert_frame_equal


def test_get_dataset():
    dataset_id = "sw5n-wg2p"
    current = nisapi.get_dataset(dataset_id, limit=10)
    expected = pl.DataFrame(None)
    assert_frame_equal(current, expected)
