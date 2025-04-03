import re

import polars as pl
import pytest

from nisapi.clean import Validate


def test_range_problems():
    """Estimate, lci, or uci outside of 0-1 should throw an error"""
    df = pl.DataFrame(
        {
            "vaccine": "flu",
            "estimate": [0.5, 1.2],
            "lci": [0.4, 1.1],
            "uci": [0.6, 1.3],
        }
    )
    v = Validate(id="test_df", df=df, mode="warn")
    # we should get an error about each col
    for col in ["estimate", "lci", "uci"]:
        matches = [re.search(f"`{col}` is not in range 0-1", x) for x in v.problems]
        # drop any None values
        matches = [x for x in matches if x is not None]
        assert len(matches) == 1


def test_range_error():
    """Range problems should throw an error if we are in error mode"""
    df = pl.DataFrame(
        {
            "vaccine": "flu",
            "estimate": [0.5, 1.2],
            "lci": [0.4, 1.1],
            "uci": [0.6, 1.3],
        }
    )
    with pytest.raises(RuntimeError):
        Validate(id="test_df", df=df, mode="error")
