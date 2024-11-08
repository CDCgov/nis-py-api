import nisapi


def test_get_datasets():
    datasets = nisapi.get_datasets()
    assert isinstance(datasets, list)

    # all datasets should be dicts
    assert all(isinstance(ds, dict) for ds in datasets)

    # all datasets should have an id
    assert all("id" in ds for ds in datasets)
