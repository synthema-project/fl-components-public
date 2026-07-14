import os
from minio import Minio


def _get_minio_client() -> Minio:
    return Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        # HTTPS in prod, plain HTTP for the dev MinIO — controlled by env.
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )


def _resolve_object_key(object_path: str) -> str:
    """Turn a catalogue-stored path into the object key within the bucket.

    The catalogue prepends the object-storage host (see data_catalogue
    ``update_use_case``), so a stored value looks like
    ``obstorageapi.k8s.synthema.rid-intrasoft.eu/AML-1_ICH_train.csv``. We strip
    that host prefix (configurable) and use the remainder as the object key; the
    bucket comes from ``MINIO_BUCKET`` and the host from ``MINIO_ENDPOINT``.
    """
    key = object_path.split("://", 1)[-1]  # drop any scheme
    host_prefix = os.getenv(
        "MINIO_CATALOGUE_HOST_PREFIX", "obstorageapi.k8s.synthema.rid-intrasoft.eu"
    ).strip("/")
    if host_prefix and key.startswith(host_prefix + "/"):
        key = key[len(host_prefix) + 1:]
    return key.lstrip("/")


def download_object(object_path: str) -> str:
    """Download a catalogue-referenced object from this node's MinIO.

    ``object_path`` example:
        ``obstorageapi.k8s.synthema.rid-intrasoft.eu/AML-1_ICH_train.csv``

    Returns the local temporary file path.
    """
    object_name = _resolve_object_key(object_path)
    bucket = os.getenv("MINIO_BUCKET", "data-annotation")

    local_filename = object_name.split("/")[-1]
    local_path = f"/tmp/{local_filename}"

    client = _get_minio_client()
    client.fget_object(bucket, object_name, local_path)

    return local_path


def ensure_bucket_exists(bucket_name: str) -> None:
    client = _get_minio_client()
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)


def upload_file(local_path: str, bucket_name: str, object_name: str) -> None:
    """Upload a local file to MinIO/ S3-compatible storage."""
    client = _get_minio_client()
    # Ensure bucket exists
    try:
        ensure_bucket_exists(bucket_name)
    except Exception:
        # If cannot create bucket, assume it exists or raise later
        pass
    client.fput_object(bucket_name, object_name, local_path)
