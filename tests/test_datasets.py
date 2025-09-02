import nisapi


def test_get_metadata():
    md = nisapi._get_metadata()
    assert isinstance(md, dict)
    assert "datasets" in md


def test_get_dataset_ids():
    ids = nisapi._get_dataset_ids()
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(id, str) for id in ids)


def test_get_dataset_metadata():
    id = "si7g-c2bs"
    assert nisapi._get_dataset_metadata(id, "id") == id
    assert isinstance(nisapi._get_dataset_metadata(id, "cleaning_arguments"), dict)
