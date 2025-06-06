from pathlib import Path

import nisapi


def test_default_cache_path():
    path = nisapi.root_cache_path()
    assert isinstance(path, Path)


def test_dataset_cache_path():
    path = nisapi.dataset_cache_path(
        root_path=Path("fake_root"), type_="raw", id="1234"
    )
    assert path == Path("fake_root", "raw", "id=1234")
