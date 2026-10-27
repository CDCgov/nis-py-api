from pathlib import Path

import pytest

from nisapi.__main__ import _build_parser


@pytest.mark.parametrize("subcommand", ["cache", "delete"])
def test_path_after_subcommand(subcommand):
    args = _build_parser().parse_args([subcommand, "--path", "custom-cache"])

    assert args.path == Path("custom-cache")


@pytest.mark.parametrize("subcommand", ["cache", "delete"])
def test_path_before_subcommand(subcommand):
    args = _build_parser().parse_args(["--path", "custom-cache", subcommand])

    assert args.path == Path("custom-cache")


def test_path_after_subcommand_overrides_path_before_subcommand():
    args = _build_parser().parse_args(
        ["--path", "first-cache", "cache", "--path", "second-cache"]
    )

    assert args.path == Path("second-cache")
