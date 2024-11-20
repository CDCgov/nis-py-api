import requests


def download_dataset_records(
    id: str,
    start_record: int,
    end_record: int,
    app_token: str = None,
    format: str = "json",
):
    """Download a specific range of rows of a data.cdc.gov dataset

    Args:
        id (str): dataset ID
        start_record (int): first row (zero-indexed)
        end_record (int): last row (zero-indexed)
        app_token (str, optional): Socrata developer app token. Defaults to None.
        format (str, optional): "json" or "csv". Defaults to "json".

    Returns:
        If format is "json", a list. If "csv", then a string
    """
    payload = {}
    if app_token is not None:
        payload["X-App-token"] = app_token

    assert end_record >= start_record
    limit = end_record - start_record + 1

    format = "json"

    url = f"https://data.cdc.gov/resource/{id}.{format}?$limit={limit}&$offset={start_record}"
    r = requests.get(url, data=payload)

    if r.status_code == 200:
        if format == "json":
            return r.json()
        else:
            return r.text
    else:
        raise RuntimeError(f"HTTP request failure: {r.status_code}")


def download_dataset_pages(id: str, page_size: int = int(1e6), app_token: str = None):
    """Download a dataset page by page

    Args:
        id (str): dataset ID
        page_size (int, optional): Page size. Defaults to 1 million.
        app_token (str, optional): Socrata developer app token. Defaults to None.

    Yields:
        Sequence of objects returned by download_dataset_records()
    """
    i = 0
    page = None
    while page != []:
        start_record = page_size * i
        end_record = start_record + page_size - 1
        page = download_dataset_records(
            id, start_record=start_record, end_record=end_record, app_token=app_token
        )
        i += 1

        if page == []:
            break
        else:
            yield page
