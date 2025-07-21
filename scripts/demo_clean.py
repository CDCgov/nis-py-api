import tempfile
from pathlib import Path

import altair as alt
import polars as pl
import yaml

import nisapi
import nisapi.clean.si7g_c2bs
from nisapi.clean import Validate

dataset_id = "si7g-c2bs"
clean_func = nisapi.clean.si7g_c2bs.clean

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

# save a copy of the partially cleaned data
tf = tempfile.NamedTemporaryFile()
clean.collect().write_parquet(tf.name)
print(f"Saved cleaned data to {tf.name}")

# this will fail until the dataset cleaning is complete
Validate(id=dataset_id, df=clean, mode="error")


def date_to_season(date: pl.Expr) -> pl.Expr:
    return (
        pl.when(date.dt.month() < 6).then(date.dt.year() - 1).otherwise(date.dt.year())
    )


alt.Chart(
    clean.filter(
        pl.col("geography_type") == pl.lit("nation"),
        pl.col("domain") == "18+ years",
        pl.col("indicator") == "received a vaccination",
    )
    .with_columns(season=date_to_season(pl.col("time_end")))
    .collect()
).encode(x="time_end", y="estimate", color="season:N", row="vaccine").mark_line().save(
    "scripts/tmp_overall.png"
)
