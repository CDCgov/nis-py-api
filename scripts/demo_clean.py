import tempfile
from pathlib import Path

import yaml

import nisapi
import nisapi.clean.ksfb_ug5d
from nisapi.clean import Validate

dataset_id = "ksfb-ug5d"
clean_func = nisapi.clean.ksfb_ug5d.clean

td = tempfile.TemporaryDirectory()

with open("scripts/secrets.yaml") as f:
    app_token = yaml.safe_load(f)["app_token"]

raw = nisapi._get_nis_raw(
    id=dataset_id,
    raw_data_path=Path(td.name),
    app_token=app_token,
)

print(f"Raw data saved to {td.name}")

# show the first few rows of the raw data
raw.head().collect().glimpse()

# try to clean the data
clean = clean_func(raw)

# look at the first few rows of the partially cleaned data
clean.head(10).collect().glimpse()
print(clean.collect().shape)

# save a copy of the partially cleaned data
tf = tempfile.NamedTemporaryFile()
clean.collect().write_parquet(tf.name)
print(f"Saved cleaned data to {tf.name}")

# this will fail until the dataset cleaning is complete
Validate(id=dataset_id, df=clean, mode="error")
