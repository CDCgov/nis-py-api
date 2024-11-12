# nis-py-api

Python API to the National Immunization Survey (NIS) data

## Data dictionary

| column                | type    | values                                     |
| --------------------- | ------- | ------------------------------------------ |
| `vaccine`             | String  | `flu`, `covid`                             |
| `geographic_level`    | String  | `nation`, `region`, `state`, `substate`    |
| `geographic_name`     | String  | `nation`, or name of the region, etc.      |
| `demographic_level`   | String  | `overall`, or varies                       |
| `demographic_name`    | String  | `overall`, or varies                       |
| `indicator_level`     | String  | always `4-level vaccination and intent`(?) |
| `indicator_name`      | String  | e.g., `received a vaccination`             |
| `week_ending`         | Date    |                                            |
| `estimate`            | Float64 | proportion between 0 and 1                 |
| `ci_half_width_95pct` | Float64 |                                            |

## Contributing

When adding a new dataset, include demonstrations that the content of the clean data is what you expected.

See also the [contributing notice](#contributing-standard-notice) below.

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
