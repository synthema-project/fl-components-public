import io
import os

import pandas as pd
import structlog
from minio import Minio
from minio.error import S3Error

from fl_server.config import DATA_CATALOGUE_URL

logger = structlog.get_logger(__name__)


# FIXME: Currently directly access to MinIO, but it must connect to web api data_catalog or model catalog
class DataCatalogueClient:
    """
    Enterprise DevSecOps S3/MinIO Client for the SYNTHEMA Data Plane.
    Handles the hermetic upload of heavy synthetic objects (Big Pools)
    and HTML validation reports to bypass MLflow artefact bloat.
    """

    def __init__(self) -> None:
        # Extract credentials securely from the environment, defaulting to standard dev credentials
        endpoint = DATA_CATALOGUE_URL.replace("http://", "").replace("https://", "")
        access_key = os.environ.get("MINIO_ACCESS_KEY", "admin")
        secret_key = os.environ.get("MINIO_SECRET_KEY", "adminpassword")

        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=DATA_CATALOGUE_URL.startswith("https")
        )
        self.bucket_name = "synthema-catalogue"
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Mathematically guarantees the target registry space exists."""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created isolated S3 bucket: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"S3/MinIO Bucket validation failed: {e}")
            raise

    def upload_synthetic_pool(self, experiment_id: str, df: pd.DataFrame) -> str:
        """Streams the Pandas DataFrame directly to MinIO, avoiding disk I/O."""
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        csv_buffer = io.BytesIO(csv_bytes)
        object_name = f"models/{experiment_id}/synthetic_big_pool.csv"

        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=csv_buffer,
            length=len(csv_bytes),
            content_type="text/csv"
        )
        logger.info(
            "Successfully pushed AML/SCD Synthetic Big Pool to S3 Data Plane",
            object_name=object_name
        )
        return object_name

    def upload_html_report(self, experiment_id: str, node_id: str, report_type: str, html_content: str) -> str:
        """Uploads the isolated utility/privacy validation HTML reports."""
        html_bytes = html_content.encode("utf-8")
        html_buffer = io.BytesIO(html_bytes)
        object_name = f"models/{experiment_id}/reports/node_{node_id}_{report_type}.html"

        self.client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=html_buffer,
            length=len(html_bytes),
            content_type="text/html"
        )
        logger.info(
            f"Successfully pushed {report_type} report for node {node_id} to S3",
            object_name=object_name
        )
        return object_name
