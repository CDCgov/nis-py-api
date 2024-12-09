import polars as pl
import yaml
import nisapi
from nisapi.clean import Validate
import nisapi.clean.vh55_3he6
import altair as alt

dataset_id = "vh55-3he6"
clean_func = nisapi.clean.vh55_3he6.clean
clean_tmp_path = "scripts/tmp_clean.parquet"

with open("scripts/secrets.yaml") as f:
    app_token = yaml.safe_load(f)["app_token"]

raw = nisapi._get_nis_raw(id=dataset_id, app_token=app_token)

# show the first few rows of the raw data
raw.head().collect().glimpse()

# try to clean the data
clean = clean_func(raw)

# look at the first few rows of the partially cleaned data
clean.head(10).collect().glimpse()

# save a copy of the partially cleaned data
clean.collect().write_parquet(clean_tmp_path)

# this will fail until the dataset cleaning is complete
Validate(id=dataset_id, df=clean)

alt.Chart(
    clean.filter(
        pl.col("geographic_type") == pl.lit("nation"),
        pl.col("demographic_type") == "overall",
        pl.col("indicator_value") == "received a vaccination",
    )
).encode(x="week_ending", y="estimate").mark_point().save("scripts/tmp_overall.png")
