import polars as pl
import yaml
import nisapi
from nisapi.clean import Validate
import nisapi.clean.udsf_9v7b
import altair as alt

dataset_id = "udsf-9v7b"

with open("scripts/secrets.yaml") as f:
    app_token = yaml.safe_load(f)["app_token"]

raw = nisapi._get_nis_raw(id=dataset_id, app_token=app_token)

raw.head().collect().glimpse()

clean = nisapi.clean.udsf_9v7b.clean(raw)
clean.head(10).collect().glimpse()

Validate(id=dataset_id, df=clean)

alt.Chart(
    clean.filter(
        pl.col("geographic_type") == pl.lit("nation"),
        pl.col("demographic_type") == "overall",
        pl.col("indicator_value") == "received a vaccination",
    )
).encode(x="week_ending", y="estimate").mark_point().save("scripts/tmp_overall.png")
