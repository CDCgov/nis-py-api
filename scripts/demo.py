import nisapi
import yaml
import polars as pl

# Load secrets from a top-level file `secrets.yaml` with key `app_token`.
# To get an app token: https://support.socrata.com/hc/en-us/articles/210138558-Generating-App-Tokens-and-API-Keys
with open("secrets.yaml") as f:
    app_token = yaml.safe_load(f)["app_token"]

# Cache all datasets. Change "skip" to "yes" if you want to rebuild the cache.
for ds in nisapi.get_datasets():
    nisapi.cache_dataset(ds["id"], overwrite="warn", app_token=app_token)

# Pull a subset of the data that's currently available
df = (
    nisapi.get_nis()
    .filter(
        # national data
        pl.col("geographic_type") == pl.lit("nation"),
        # by age group
        pl.col("demographic_type") == pl.lit("age"),
        # showing %vaccinated through time
        pl.col("indicator_value") == pl.lit("received a vaccination"),
    )
    .drop(
        "geographic_type",
        "geographic_value",
        "demographic_type",
        "indicator_type",
        "indicator_value",
    )
    .collect()
)

# Print the first few rows
print(df)