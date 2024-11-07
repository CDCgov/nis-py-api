from sodapy import Socrata
import polars as pl
from datetime import date


def _get_dataset_ids(disease: str, start_date: date, end_date: date) -> [str]:
    return "udsf-9v7b"


def get_dataset(id: str, app_token=None, **kwargs) -> pl.DataFrame:
    with Socrata("data.cdc.gov", app_token) as client:
        results = client.get(id, **kwargs)

    return pl.DataFrame(results)


def get_nis(disease: str, start_date: date, end_date: date, app_token=None, **kwargs):
    dataset_ids = _get_dataset_ids(
        disease=disease, start_date=start_date, end_date=end_date
    )

    return pl.concat([get_dataset(id) for id in dataset_ids])
