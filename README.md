# nis-py-api

Python API to access parts of the [National Immunization Survey](https://www.cdc.gov/nis/about/index.html) (NIS) data available via <https://data.cdc.gov/>. This package helps with data download, cleaning, and caching, ensuring quick access to data that have consistent field names and certain validations.

## Getting started

- This a poetry-enabled project. Use `poetry install` to install.
- Get a [Socrata app token](https://support.socrata.com/hc/en-us/articles/210138558-Generating-App-Tokens-and-API-Keys). Copy `scripts/secrets_template.yaml` to `scripts/secrets.yaml` and fill out the `app_token`.
- See `scripts/demo.py` for an example of how to cache and query the data:
  - `python -m nisapi cache app_token={SOCRATA_APP_TOKEN}` or `nisapi.cache_all_datasets(app_token={SOCRATA_APP_TOKEN})` to download, clean, and cache data
  - `nisapi.get_nis()` to get a lazy data frame pointing to that locally cached, clean data
  - `python -m nisapi delete` or `nisapi.delete_cache()` to delete the cache, if needed
- See `scripts/demo_clean.py` for an example of a script that you could run while iteratively developing the cleaning code in `nisapi/clean/`.
- See `scripts/demo_cloud.py` for a demo of how the data could be downloaded, cleaned, uploaded to Azure Blob Storage, and then downloaded from there. You will need to fill out the `azure:` keys in `secrets.yaml`.
- Run `streamlit run scripts/demo_streamlit.py` to quickly query and visualize the data with a [streamlit](https://streamlit.io/) app.

## Data dictionary

The data have these columns, in order, with these types:

| column           | type    |
| ---------------- | ------- |
| `vaccine`        | String  |
| `geography_type` | String  |
| `geography`      | String  |
| `domain_type`    | String  |
| `domain`         | String  |
| `indicator_type` | String  |
| `indicator`      | String  |
| `time_type`      | String  |
| `time_start`     | Date    |
| `time_end`       | Date    |
| `estimate`       | Float64 |
| `lci`            | Float64 |
| `uci`            | Float64 |

Note the pairs `geography_type` and `geography`, `domain_type` and `domain`, and `indicator_type` and `indicator`.

Rows that were suppressed in the raw data are dropped. This includes data with suppression flag `"1"`, indicating small sample size, and data with flag `"."`, which may indicate that data were not collected.

### `vaccine`

- One of `"flu"` or `"covid"`

### `geography_type`

- One of `"nation"`, `"region"`, `"admin1"`, `"substate"`, `"county"`
- "Region" means HHS Region
- First-level administrative divisions (`"admin1"`) include US states, territories, and the District of Columbia
- "Substate" includes:
  - Chicago, and the rest of Illinois
  - New York City, and the rest of New York
  - Philadelphia, and the rest of Pennsylvania
  - Bexar County, City of Houston, and the rest of Texas

### `geography`

- If `geography_type` is `"nation"`, then this is `"nation"`
- If `"region"`, then a string of the form `"Region 1"`
- If `"admin1"`, then the full name of the jurisdiction
- If `"substate"`, no validation is currently applied
- If `"county"`, then the 5-digit FIPS code

### `domain_type`

- There are multiple types, including `"age"`

### `domain`

- If `domain_type` is `"age"`, then this is the age group, with the form `"x-y years"`, `"x-y months"`, `"x+ years"`, `"x+ months"`, or `"x months-y years"`

### `indicator_type`

- In newer data, this is always `"4-level vaccination and intent"`
- In historical COVID-19 and flu data, there are a wide range of indicators

### `indicator`

- The value of the indicator, e.g., `"received a vaccination"`

### `time_type`

- One of `"month"` or `"week"`

### `time_start` and `time_end`

- Period of time associated with the observation. Note that "monthly" and "weekly" observations do not always align with calendar weeks or months, so we specify the two dates explicitly.
- Certain assumptions were made about `time_start` dates for datasets that provided "week ending" dates.
- Time start is always before time end.

### `estimate`

- Proportion (i.e., a number between 0 and 1) of the population (defined by geography and domain) that has the characteristic described by the indicator

### `lci` and `uci`

- The lower and upper limits of the 95% confidence interval, measured in the same units as `estimate`
- Confidence interval always bracket the `estimate`

## Contributing

### Adding a new dataset

1. Find the dataset you want to clean on <https://data.cdc.gov/>.
2. Add the dataset to `nisapi/datasets.yaml`.
   - At a minimum, you must include the dataset ID.
   - It is helpful to also include URL, vaccine, date range, and universe.
3. Create a dataset-specific module in `nisapi/clean/`. It should have a main function `clean()`.
   - Start with a `clean()` function that does nothing and just returns the input data frame.
4. Add the `import` and `elif` statements for this dataset ID to `clean_dataset()` in `nisapi/clean/__init__.py`.
5. Run `scripts/clean_demo.py`. This should cache the raw dataset, run the cleaning function, and fail on validation.
6. Iteratively update the dataset-specific `clean()` function until validation passes.
   - Ideally, `clean()` should be a series of pipe functions.
   - If a cleaning step is specific to a single dataset, keep that in the dataset-specific submodule. If a step is shared between datasets, move it into `helpers.py`.
   - If multiple indicators are redundant, validate that redundancy in code, and then pick only one indicator. (E.g., `ksfb-ug5d` and `sw5n-wg2p` drop the up-to-date indicator in favor of the 4-level vaccination intent indicator.)
   - If you find some dataset-specific anomaly or validation problem, make a note of it in `datasets.yaml`.
7. Open a PR.
   - Include any validations if you needed to correct an anomaly.

## Project Admin

- Scott Olesen <ulp7@cdc.gov> (CDC/CFA)

---

## General Disclaimer

This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm). GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise.

## Public Domain Standard Notice

This repository constitutes a work of the United States Government and is not subject to domestic copyright protection under 17 USC § 105. This repository is in the public domain within the United States, and copyright and related rights in the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/). All contributions to this repository will be released under the CC0 dedication. By submitting a pull request you are agreeing to comply with this waiver of copyright interest.

## License Standard Notice

This repository is licensed under [ASL v2](http://www.apache.org/licenses/LICENSE-2.0.html) or later.

## Privacy Standard Notice

This repository contains only non-sensitive, publicly available data and information. All material and community participation is covered by the [Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md) and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md). For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice

This repository is not a source of government records but is a copy to increase collaboration and collaborative potential. All government records will be published through the [CDC web site](http://www.cdc.gov).

## Standard Notice

Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo) and submitting a pull request. (If you are new to GitHub, you might start with a [basic tutorial](https://help.github.com/articles/set-up-git).) By contributing to this project, you grant a world-wide, royalty-free, perpetual, irrevocable, non-exclusive, transferable license to all users under the terms of the [Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or later.

All comments, messages, pull requests, and other submissions received through CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).
