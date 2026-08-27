import io
import json
import os
from urllib.parse import urlparse
from minio import Minio
from pandas import DataFrame


def _get_minio_client() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT")
    if not endpoint:
        raise RuntimeError("MINIO_ENDPOINT must be configured")
    endpoint = endpoint.removeprefix("http://").removeprefix("https://")
    return Minio(
        endpoint=endpoint,
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=os.getenv("MINIO_SECURE", "true").lower() == "true",
    )


def upload_dataframe(data: DataFrame, *, use_case: str, model_name: str, node_name: str) -> str:
    """Upload generated data and return its stable object-storage URI."""
    bucket = os.getenv("MINIO_SYNTHETIC_BUCKET", "synthetic-data-for-inference")
    key = f"{use_case}/{model_name}/{node_name}/synthetic_data.csv"
    payload = data.to_csv(index=False).encode("utf-8")
    client = _get_minio_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    client.put_object(bucket, key, io.BytesIO(payload), length=len(payload), content_type="text/csv")
    scheme = "https" if os.getenv("MINIO_SECURE", "true").lower() == "true" else "http"
    return f"{scheme}://{os.getenv('MINIO_ENDPOINT')}/{bucket}/{key}"


def upload_json(payload: dict, *, use_case: str, model_name: str, node_name: str, filename: str) -> str:
    """Upload a JSON contract beside generated data and return its object URI."""
    bucket = os.getenv("MINIO_SYNTHETIC_BUCKET", "synthetic-data-for-inference")
    key = f"{use_case}/{model_name}/{node_name}/{filename}"
    encoded = json.dumps(payload, indent=2).encode("utf-8")
    client = _get_minio_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    client.put_object(bucket, key, io.BytesIO(encoded), length=len(encoded), content_type="application/json")
    scheme = "https" if os.getenv("MINIO_SECURE", "true").lower() == "true" else "http"
    return f"{scheme}://{os.getenv('MINIO_ENDPOINT')}/{bucket}/{key}"


def upload_html(html: str, *, use_case: str, model_name: str, node_name: str,
               filename: str) -> str:
    """Upload a validation report; the validator may later be supplied privately."""
    bucket = os.getenv("MINIO_SYNTHETIC_BUCKET", "synthetic-data-for-inference")
    key = f"{use_case}/{model_name}/{node_name}/{filename}"
    payload = html.encode("utf-8")
    client = _get_minio_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    client.put_object(bucket, key, io.BytesIO(payload), length=len(payload), content_type="text/html")
    scheme = "https" if os.getenv("MINIO_SECURE", "true").lower() == "true" else "http"
    return f"{scheme}://{os.getenv('MINIO_ENDPOINT')}/{bucket}/{key}"


def download_object(object_path: str) -> str:
    """
    object_path example:
    obstorageapi.k8s.synthema.rid-intrasoft.eu/aml1_NODE1_file.csv

    Returns local temporary file path.
    """

    parsed = urlparse(object_path if "://" in object_path else f"https://{object_path}")

    configured_endpoint = (os.getenv("MINIO_ENDPOINT") or "").removeprefix("http://").removeprefix("https://")
    bucket = parsed.netloc
    if parsed.scheme == "s3":
        bucket = parsed.netloc
    elif configured_endpoint:
        # The catalogue's public URL identifies the MinIO gateway, not the
        # bucket.  Dataset objects are all stored in the ingestion bucket.
        bucket = os.getenv("MINIO_BUCKET_NAME", "data-annotation")
    object_name = parsed.path.lstrip("/")

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
