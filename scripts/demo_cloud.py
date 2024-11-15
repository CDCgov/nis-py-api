import yaml
import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from pathlib import Path
import nisapi
import tempfile
import polars as pl


def get_client(client_id, storage_account_name) -> BlobServiceClient:
    os.environ["AZURE_CLIENT_ID"] = client_id

    account_url = f"https://{storage_account_name}.blob.core.windows.net"
    default_credential = DefaultAzureCredential()
    return BlobServiceClient(account_url, credential=default_credential)


def container_exists(client: BlobServiceClient, container_id: str) -> bool:
    return container_id in [c["name"] for c in client.list_containers()]


def upload_blobs(
    client: BlobServiceClient,
    container_id: str,
    blob_root: str,
    local_dir: Path,
    overwrite: str = "skip",
) -> None:
    for dirpath, dirnames, filenames in os.walk(local_dir):
        for f in filenames:
            blob_id = get_blob_id(blob_root, local_dir, Path(dirpath, f))

            blob_client = client.get_blob_client(container=container_id, blob=blob_id)

            if blob_client.exists() and overwrite == "skip":
                print(f"Skipping {blob_id=}")
            else:
                print(f"Uploading {blob_id=}")
                local_path = Path(local_dir, dirpath, f)
                with open(local_path, "rb") as data:
                    blob_client.upload_blob(data)


def download_blobs(
    client: BlobServiceClient, container_id: str, blob_root: str, local_dir: Path
) -> None:
    container_client = client.get_container_client(container_id)
    for blob in container_client.list_blobs(name_starts_with=blob_root):
        blob_client = container_client.get_blob_client(blob)
        local_path = Path(local_dir, blob.name)

        # ensure the parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "wb") as data:
            blob_client.download_blob().readinto(data)


def get_blob_id(blob_root: str, local_root: Path, local_path: Path) -> str:
    return str(Path(blob_root, local_path.relative_to(local_root)))


assert (
    get_blob_id(
        "nis",
        Path("/home/ulp7/.cache/nisapi/"),
        Path("/home/ulp7/.cache/nisapi/id=foo/bar.csv"),
    )
    == "nis/id=foo/bar.csv"
)


# set service principal
with open("scripts/secrets.yaml") as f:
    secrets = yaml.safe_load(f)

client = get_client(
    secrets["azure"]["client_id"], secrets["azure"]["storage_account_name"]
)

# upload the blobs
print("Uploading blobs")
upload_blobs(
    client=client,
    container_id=secrets["azure"]["container_id"],
    blob_root=secrets["azure"]["blob_root"],
    local_dir=nisapi.default_cache_path(),
)

# download the file to a new location
print("Downloading blobs")
with tempfile.TemporaryDirectory() as tmpdir:
    data_path = Path(tmpdir, "nis")
    download_blobs(
        client=client,
        container_id=secrets["azure"]["container_id"],
        blob_root=secrets["azure"]["blob_root"],
        local_dir=data_path,
    )

    df = pl.scan_parquet(str(data_path))
    print(df.head().collect())
