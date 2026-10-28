import polars as pl
import pytest
from polars.testing import assert_frame_equal

import nisapi


@pytest.mark.parametrize(
    ("app_token", "expected_app_token"),
    [("test-token", "test-token"), (None, None)],
)
def test_download_dataset_uses_cfasodapy(
    monkeypatch, app_token, expected_app_token
):
    call = {}

    class Query:
        def __init__(self, **kwargs):
            call.update(kwargs)

        def get_pages(self, page_size):
            call["page_size"] = page_size
            return iter([[{"id": 1}, {"id": 2}], [{"id": 3}]])

    monkeypatch.setattr(nisapi.cfasodapy, "Query", Query)

    actual = nisapi._download_dataset(id="abcd-1234", app_token=app_token)

    assert call == {
        "domain": "data.cdc.gov",
        "id": "abcd-1234",
        "app_token": expected_app_token,
        "page_size": 100_000,
    }
    assert_frame_equal(
        actual,
        pl.DataFrame({"id": [1, 2, 3]}),
    )
