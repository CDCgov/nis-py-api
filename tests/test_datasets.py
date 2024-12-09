import nisapi


def test_get_datasets():
    ids = nisapi._get_dataset_ids()
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(id, str) for id in ids)
