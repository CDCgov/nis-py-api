import requests
import math
from typing import Sequence

domain = "data.cdc.gov"


def n_dataset_rows(id: str, app_token: str = None, domain: str = domain) -> int:
    url = f"https://{domain}/resource/{id}.json?$select=count(:id)"
    r = _get_request(url, app_token=app_token)

    result = r.json()
    assert len(result) == 1
    assert "count_id" in result[0]
    return int(result[0]["count_id"])


def download_dataset_records(
    id: str,
    start_record: int,
    end_record: int,
    app_token: str = None,
    domain: str = domain,
) -> list[dict]:
    """Download a specific range of rows of a data.cdc.gov dataset

    Args:
        id (str): dataset ID
        start_record (int): first row (zero-indexed)
        end_record (int): last row (zero-indexed)
        app_token (str, optional): Socrata developer app token. Defaults to None.
        domain (str, optional): defaults to "data.cdc.gov"

    Returns:
        If format is "json", a list. If "csv", then a string
    """

    assert end_record >= start_record
    limit = end_record - start_record + 1

    url = f"https://{domain}/resource/{id}.json?$limit={limit}&$offset={start_record}&$order=:id"
    r = _get_request(url, app_token=app_token)

    return r.json()


def _get_request(url: str, app_token: str = None) -> requests.Request:
    payload = {}
    if app_token is not None:
        payload["X-App-token"] = app_token

    r = requests.get(url, data=payload)
    if r.status_code == 200:
        return r
    else:
        raise RuntimeError(
            f"HTTP request failure: url '{url}' failed with code {r.status_code}"
        )


def download_dataset_pages(
    id: str, page_size: int = int(1e5), app_token: str = None, verbose: bool = True
) -> Sequence[list[dict]]:
    """Download a dataset page by page

    Args:
        id (str): dataset ID
        page_size (int, optional): Page size. Defaults to 1 million.
        app_token (str, optional): Socrata developer app token. Defaults to None.
        verbose (bool): If True (default), print progress

    Yields:
        Sequence of objects returned by download_dataset_records()
    """
    n_rows = n_dataset_rows(id, app_token=app_token)
    n_pages = math.ceil(n_rows / page_size)

    if verbose:
        print(
            f"Downloading dataset {id=}: {n_rows} rows in {n_pages} page(s) of {page_size} rows each"
        )

    for i in range(n_pages):
        if verbose:
            print(f"  Downloading page {i + 1}/{n_pages}")

        start_record = i * page_size
        end_record = (i + 1) * page_size - 1
        page = download_dataset_records(
            id, start_record=start_record, end_record=end_record, app_token=app_token
        )

        assert len(page) > 0
        assert len(page) <= page_size

        yield page
