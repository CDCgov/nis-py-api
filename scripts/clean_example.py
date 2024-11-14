import polars as pl
import os.path
import yaml
import nisapi

tmp_path = "tmp_raw_dataset.parquet"
dataset_id = "sw5n-wg2p"

if not os.path.exists(tmp_path):
    with open("secrets.yaml") as f:
        app_token = yaml.safe_load(f)["app_token"]

    nisapi.download_dataset(dataset_id, app_token=app_token).write_parquet(tmp_path)

df = pl.read_parquet(tmp_path)

clean = nisapi.clean_dataset(dataset_id, df)
clean.glimpse()
clean.tail().glimpse()
