import os
from urllib.parse import urlparse
from minio import Minio


def _get_minio_client() -> Minio:
    return Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=True,
    )


def download_object(object_path: str) -> str:
    """
    object_path example:
    obstorageapi.k8s.synthema.rid-intrasoft.eu/aml1_NODE1_file.csv

    Returns local temporary file path.
    """

    # Add https:// so urlparse works correctly
    parsed = urlparse(f"https://{object_path}")

    bucket = parsed.netloc
    object_name = parsed.path.lstrip("/")

    local_filename = object_name.split("/")[-1]
    local_path = f"/tmp/{local_filename}"

    client = _get_minio_client()

    client.fget_object(bucket, object_name, local_path)

    return local_path
