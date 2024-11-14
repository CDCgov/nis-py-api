from nisapi.clean import valid_age_groups
import polars as pl


def test_validate_age_groups():
    assert valid_age_groups(pl.Series(["18-49 years"]))

    assert valid_age_groups(pl.Series(["65+ years"]))

    assert valid_age_groups(pl.Series(["18-49 years", "50-64 years", "65+ years"]))

    # en dash should fail
    assert not valid_age_groups(pl.Series(["18–49 years"]))

    # missing "years" should fail
    assert not valid_age_groups(pl.Series(["18-49"]))

    # fail if there are spaces
    assert not valid_age_groups(pl.Series(["18 - 49 years"]))
