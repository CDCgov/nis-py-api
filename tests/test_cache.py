from pathlib import Path

import nisapi


def test_default_cache_path():
    path = nisapi.default_cache_path()
    assert isinstance(path, Path)


def test_dataset_cache_path():
    path = nisapi.get_data_path(type_="raw", id="1234", path=Path("fake_root"))
    assert path == Path("fake_root", "raw", "id=1234")
