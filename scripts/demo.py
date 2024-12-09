import polars as pl
import yaml

import nisapi

# Load secrets from a top-level file `secrets.yaml` with key `app_token`.
with open("scripts/secrets.yaml") as f:
    app_token = yaml.safe_load(f)["app_token"]

# Clear and rebuild
# nisapi.delete_cache()
nisapi.cache_all_datasets(app_token=app_token)

# Pull a subset of the data that's currently available
(
    nisapi.get_nis()
    .filter(
        # national data
        pl.col("geographic_type") == pl.lit("nation"),
        # by age group
        pl.col("domain_type") == pl.lit("age"),
        # showing %vaccinated through time
        pl.col("indicator_value") == pl.lit("received a vaccination"),
    )
    # get the first few rows
    .head(10)
    .collect()
    .glimpse()
)
