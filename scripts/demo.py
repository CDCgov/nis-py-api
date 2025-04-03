import os

import polars as pl

import nisapi

# Clear cache if needed
# nisapi.delete_cache()

# Check that the app token is present
if "SOCRATA_APP_TOKEN" not in os.environ:
    raise ValueError("SOCRATA_APP_TOKEN environment variable not found")

nisapi.cache_all_datasets(app_token=os.environ["SOCRATA_APP_TOKEN"])

# Pull a subset of the data that's currently available
(
    nisapi.get_nis()
    .filter(
        # national data
        pl.col("geography_type") == pl.lit("nation"),
        # by age group
        pl.col("domain_type") == pl.lit("age"),
        # showing %vaccinated through time
        pl.col("indicator") == pl.lit("received a vaccination"),
    )
    # get the first few rows
    .head(10)
    .collect()
    .glimpse()
)
