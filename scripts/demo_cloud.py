import yaml
import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient
from pathlib import Path
import nisapi
import tempfile


def get_blob_client(config, verbose=True) -> BlobClient:
    os.environ["AZURE_CLIENT_ID"] = config["client_id"]

    storage_account_name = config["storage_account_name"]
    account_url = f"https://{storage_account_name}.blob.core.windows.net"
    default_credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url, credential=default_credential)

    container_id = config["container_id"]
    assert container_id in blob_service_client.list_containers()
    container_client = blob_service_client.get_container_client(container_id)

    blob_id = config["blob_id"]

    if verbose:
        print(
            f"Blob {blob_id} already in container {container_id}?",
            blob_id in container_client.list_blobs(),
        )

    blob_client = blob_service_client.get_blob_client(
        container=container_id, blob=blob_id
    )

    return blob_client


def upload_blob(blob_client: BlobClient, path: Path) -> None:
    with open(path, "rb") as data:
        blob_client.upload_blob(data)


def download_blob(blob_client: BlobClient, path: Path) -> None:
    with open(path, "wb") as data:
        blob_client.download_blob().readinto(data)


# set service principal
with open("scripts/secrets.yaml") as f:
    secrets = yaml.safe_load(f)

blob_client = get_blob_client(secrets["azure"])

# upload a file
upload_blob(blob_client, nisapi.default_cache_path())

# download the file to a new location
with tempfile.TemporaryDirectory() as tmpdir:
    data_path = Path(tmpdir, "nis")
    download_blob(blob_client, data_path)

    for path, dirs, files in os.walk(data_path):
        print(path)
        for f in files:
            print(f)
