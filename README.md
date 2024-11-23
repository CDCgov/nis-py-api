# nis-py-api

Python API to the National Immunization Survey (NIS) data.

:construction: This tool is in alpha development. The API and data schema are not stable.

## Getting started

- This a poetry-enabled project.
- Get a [Socrata app token](https://support.socrata.com/hc/en-us/articles/210138558-Generating-App-Tokens-and-API-Keys). Copy `scripts/secrets_template.yaml` to `scripts/secrets.yaml` and fill out the `app_token`.
- See `scripts/demo.py` for an example of how to cache and query the data:
  - `nisapi.cache_all_datasets()` to download, clean, and cache data
  - `nisapi.get_nis()` to get a lazy data frame pointing to that locally cached, clean data
  - `nisapi.delete_cache()` to clear the cache, if needed
- See `scripts/demo_clean.py` for an example of a script that you could run while iteratively developing the cleaning code in `nisapi/clean/`.
- See `scripts/demo_cloud.py` for a demo of how the data could be downloaded, cleaned, uploaded to Azure Blob Storage, and then downloaded from there. You will need to fill out the `azure:` keys in `secrets.yaml`.

## Data dictionary

The data have these columns, in order, with these types:

| column              | type    |
| ------------------- | ------- |
| `vaccine`           | String  |
| `geographic_type`   | String  |
| `geographic_value`  | String  |
| `demographic_type`  | String  |
| `demographic_value` | String  |
| `indicator_type`    | String  |
| `indicator_value`   | String  |
| `time_type`         | String  |
| `time_start`        | Date    |
| `time_end`          | Date    |
| `estimate`          | Float64 |
| `lci`               | Float64 |
| `uci`               | Float64 |

Note the paired use of "type" and "value" columns.

Rows that were suppressed in the raw data are dropped. This includes data with suppression flag `"1"`, indicating small sample size, and data with flag `"."`, which may indicate that data were not collected.

### `vaccine`

- One of `"flu"` or `"covid"`

### `geographic_type`

- One of `"nation"`, `"region"`, `"admin1"`, `"substate"`
- "Region" means HHS Region
- First-level administrative divisions include US states, territories, and the District of Columbia

### `geographic_value`

- If `geographic_type` is `"nation"`, then this is `"nation"`
- If `"region"`, then a string of the form `"Region 1"`
- If `"admin1"`, then the full name of the jurisdiction
- If `"substate"`, no validation is currently applied

### `demographic_type`

- There are multiple types, including `"overall"` and `"age"`
- Note that "overall" might refer only to certain age groups (e.g., 18+)

### `demographic_value`

- If `demographic_type` is `"overall"`, then this is `"overall"`
- If `demographic_type` is `"age"`, then this is the age group, with the form `"x-y years"` or `"x+ years"`

### `indicator_type`

- Always `"4-level vaccination and intent"`

### `indicator_value`

- The value of the indicator, e.g., `"received a vaccination"`

### `time_type`

- One of `"monthly"` or `"weekly"`

### `time_start` and `time_end`

Period of time associated with the observation. Note that "monthly" and "weekly" observations do not always align with calendar weeks or months, so we specify the two dates explicitly.

### `estimate`

- Proportion (i.e., a number between 0 and 1) of the population (defined by geography and demography) that has the characteristic described by the indicator

### `lci` and `uci`

The lower and upper limits of the 95% confidence interval, measured in the same units as `estimate`

## Contributing

See also the [contributing notice](#contributing-standard-notice) below.

### Adding a new dataset

1. Annotate the dataset ID and URL in `datasets.yaml`
2. Use a script like `scripts/demo_clean.py` to iterate when formulating the cleaning steps.

When adding a new dataset, include demonstrations that the content of the clean data is what you expected.

## Project Admin

- Scott Olesen <ulp7@cdc.gov> (CDC/CFA)

## General Disclaimer

This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm). GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise.

## Public Domain Standard Notice

This repository constitutes a work of the United States Government and is not subject to domestic copyright protection under 17 USC § 105. This repository is in the public domain within the United States, and copyright and related rights in the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/). All contributions to this repository will be released under the CC0 dedication. By submitting a pull request you are agreeing to comply with this waiver of copyright interest.

## License Standard Notice

This repository is licensed under [ASL v2](http://www.apache.org/licenses/LICENSE-2.0.html) or later.

## Privacy Standard Notice

This repository contains only non-sensitive, publicly available data and information. All material and community participation is covered by the [Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md) and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md). For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice

Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo) and submitting a pull request. (If you are new to GitHub, you might start with a [basic tutorial](https://help.github.com/articles/set-up-git).) By contributing to this project, you grant a world-wide, royalty-free, perpetual, irrevocable, non-exclusive, transferable license to all users under the terms of the [Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or later.

All comments, messages, pull requests, and other submissions received through CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice

This repository is not a source of government records but is a copy to increase collaboration and collaborative potential. All government records will be published through the [CDC web site](http://www.cdc.gov).
