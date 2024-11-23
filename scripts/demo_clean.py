import polars as pl
import os.path
import yaml
import nisapi
import altair as alt

tmp_path = "scripts/tmp_nis.parquet"
dataset_id = "udsf-9v7b"

if not os.path.exists(tmp_path):
    with open("scripts/secrets.yaml") as f:
        app_token = yaml.safe_load(f)["app_token"]

    nisapi.download_dataset(dataset_id, app_token=app_token).write_parquet(tmp_path)

df = pl.read_parquet(tmp_path)

df.glimpse()
df.tail().glimpse()

clean = nisapi.clean_dataset(dataset_id, df)
clean.glimpse()
clean.tail().glimpse()

alt.Chart(
    clean.filter(
        pl.col("geographic_type") == pl.lit("nation"),
        pl.col("demographic_type") == "overall",
        pl.col("indicator_value") == "received a vaccination",
    )
).encode(x="week_ending", y="estimate").mark_point().save("scripts/tmp_overall.png")
