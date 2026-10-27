import json
from copy import deepcopy
from pathlib import Path

import jsonschema
import pytest

METADATA_PATH = Path(__file__).parents[1] / "nisapi" / "metadata.json"
SCHEMA_PATH = Path(__file__).parents[1] / "nisapi" / "metadata.schema.json"


def _read_json(path: Path):
    with path.open() as file:
        return json.load(file)


def test_metadata_matches_schema():
    jsonschema.validate(
        instance=_read_json(METADATA_PATH),
        schema=_read_json(SCHEMA_PATH),
    )


def test_metadata_schema_rejects_unknown_cleaning_method():
    metadata = deepcopy(_read_json(METADATA_PATH))
    metadata["datasets"][0]["cleaning_arguments"]["clean_unknown"] = {}

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=metadata, schema=_read_json(SCHEMA_PATH))
