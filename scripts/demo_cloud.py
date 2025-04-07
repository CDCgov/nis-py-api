import os
import tempfile
from pathlib import Path

import polars as pl
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

import nisapi


def get_client(client_id: str, storage_account_name: str) -> BlobServiceClient:
    """Create a blob service client

    Args:
        client_id (string): service principal
        storage_account_name (str): storage account name

    Returns:
        BlobServiceClient: with access to the storage account
    """
    os.environ["AZURE_CLIENT_ID"] = client_id

    account_url = f"https://{storage_account_name}.blob.core.windows.net"
    default_credential = DefaultAzureCredential()
    return BlobServiceClient(account_url, credential=default_credential)


def upload_blobs(
    client: BlobServiceClient,
    container_id: str,
    blob_root: str,
    local_dir: Path,
    overwrite: str = "skip",
) -> None:
    """Upload a file tree to blobs

    Args:
        client (BlobServiceClient): authenticated service client
        container_id (str): blob storage container ID
        blob_root (str): prefix of names for all the blobs, to copy files to
        local_dir (Path): local directory to copy files from
        overwrite (str, optional): If `"skip"` (default), then do not overwrite
          existing blobs. Otherwise, overwrite.
    """
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
    """Download blobs to a local file tree

    Args:
        client (BlobServiceClient): authenticated service client
        container_id (str): blob storage container ID
        blob_root (str): Prefix of blob names. All blobs with this prefix will be
          downloaded to `local_dir`.
        local_dir (Path): local path for blobs to be downloaded to
    """
    container_client = client.get_container_client(container_id)
    for blob in container_client.list_blobs(name_starts_with=blob_root):
        blob_client = container_client.get_blob_client(blob)
        local_path = Path(local_dir, blob.name)

        # ensure the parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "wb") as data:
            blob_client.download_blob().readinto(data)


def get_blob_id(blob_root: str, local_root: Path, local_path: Path) -> str:
    """Convert a local path to a remote blob ID

    Args:
        blob_root (str): blob ID prefix
        local_root (Path): generate blob ID using file paths relative to this path
        local_path (Path): local file path

    Returns:
        str: blob ID
    """
    return str(Path(blob_root, local_path.relative_to(local_root)))


# ad hoc test for get_blob_id()
assert (
    get_blob_id(
        "nis",
        Path("/home/ulp7/.cache/nisapi/"),
        Path("/home/ulp7/.cache/nisapi/id=foo/bar.csv"),
    )
    == "nis/id=foo/bar.csv"
)

# set up and authenticate the client
client = get_client(
    os.environ["AZURE_CLIENT_ID"], os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
)

# upload the blobs from local storage to remote, to demonstrate how a definitive
# dataset would be placed in common storage
print("Uploading blobs")
upload_blobs(
    client=client,
    container_id=os.environ["AZURE_CONTAINER_ID"],
    blob_root=os.environ["AZURE_BLOB_ROOT"],
    local_dir=nisapi._root_cache_path(),
)

# download the file to a new local directory, to demonstrate how another user
# could download the definitive data locally
print("Downloading blobs")
with tempfile.TemporaryDirectory() as tmpdir:
    download_blobs(
        client=client,
        container_id=os.environ["AZURE_CONTAINER_ID"],
        blob_root=os.environ["AZURE_BLOB_ROOT"],
        local_dir=tmpdir,
    )

    data_path = Path(tmpdir, "nis", "clean")

    # print the downloaded data, to show it's accessible
    df = pl.scan_parquet(str(data_path))
    print(df.head().collect())
    print("Data shape:", df.collect().shape)
