import re
from datetime import date

import polars as pl
import pytest

from nisapi.clean import Validate
from nisapi.clean.helpers import data_schema

BASE_DF = pl.DataFrame(
    {
        "vaccine": "flu",
        "estimate": [0.5, 0.6],
        "lci": [0.4, 0.5],
        "uci": [0.6, 0.7],
        "geography_type": "nation",
        "geography": "nation",
        "domain_type": "all",
        "domain": "all",
        "indicator_type": "vaccination",
        "indicator": "received a vaccination",
        "time_type": "week",
        "time_start": [date(2025, 8, 14), date(2025, 8, 20)],
        "time_end": [date(2025, 8, 21), date(2025, 8, 27)],
        "sample_size": 10,
    },
    schema=data_schema,
)

BAD_RANGE_DF = BASE_DF.with_columns(
    estimate=pl.Series([0.5, 1.2]),
    lci=pl.Series([0.4, 1.1]),
    uci=pl.Series([0.6, 1.3]),
)


def test_range_problems():
    """Estimate, lci, or uci outside of 0-1 should throw an error"""
    v = Validate(id="test_df", df=BAD_RANGE_DF, mode="warn")
    # we should get an error about each col
    for col in ["estimate", "lci", "uci"]:
        matches = [re.search(f"`{col}` is not in range 0-1", x) for x in v.problems]
        # drop any None values
        matches = [x for x in matches if x is not None]
        assert len(matches) == 1


def test_range_error():
    """Range problems should throw an error if we are in error mode"""
    with pytest.raises(RuntimeError):
        Validate(id="test_df", df=BAD_RANGE_DF, mode="error")


def test_has_excess_whitespace():
    bad_strings = pl.Series(
        [
            "non-space\twhitespace",
            "multiple  whitespace",
            " starting whitespace",
            "trailing whitespace ",
        ]
    )
    assert Validate._has_excess_whitespace(bad_strings).all()

    assert Validate._has_excess_whitespace(pl.Series(["this is ok"])).not_().all()


def test_has_excess_whitespace_df():
    df = BASE_DF.with_columns(indicator=pl.lit("received  a  vaccination")).select(
        BASE_DF.columns
    )

    v = Validate(id="test_df", df=df, mode="warn")
    assert len(v.problems) == 1
    assert "whitespace" in v.problems[0]


def test_has_bad_capitalization():
    assert Validate._has_bad_capitalization(
        pl.Series(["I got a vaccine", "COVID-19"])
    ).all()
    assert (
        Validate._has_bad_capitalization(pl.Series(["no problems here"])).not_().all()
    )
