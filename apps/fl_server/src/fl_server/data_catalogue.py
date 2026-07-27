import io
import os
import time
from typing import Dict, List, Optional

import pandas as pd
import requests
import structlog
from minio import Minio
from minio.error import S3Error

from fl_server.config import DATA_CATALOGUE_URL

logger = structlog.get_logger(__name__)


# --- Keycloak service-account auth for the catalogue -------------------------
# The fl-server authenticates to the (Keycloak-protected) data-catalogue with a
# service-account (client_credentials) token, the same way the fl-client reaches
# the restapi. Falls back to a static CATALOGUE_TOKEN when Keycloak is unset.
_KEYCLOAK_SERVER_URL = os.getenv("KEYCLOAK_SERVER_URL")
_KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM_NAME")
_KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
_KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")

_service_token: Optional[str] = None
_service_token_exp: float = 0.0


def _get_catalogue_token() -> Optional[str]:
    """Return a bearer token for the catalogue API.

    Prefers a cached Keycloak service-account access token (client_credentials),
    refreshing it shortly before expiry. When the Keycloak service account is not
    configured, falls back to a static ``CATALOGUE_TOKEN`` (or None).
    """
    global _service_token, _service_token_exp
    if not (
        _KEYCLOAK_SERVER_URL
        and _KEYCLOAK_REALM
        and _KEYCLOAK_CLIENT_ID
        and _KEYCLOAK_CLIENT_SECRET
    ):
        return os.getenv("CATALOGUE_TOKEN")

    now = time.time()
    if _service_token and now < _service_token_exp - 30:
        return _service_token

    response = requests.post(
        f"{_KEYCLOAK_SERVER_URL}/realms/{_KEYCLOAK_REALM}/protocol/openid-connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": _KEYCLOAK_CLIENT_ID,
            "client_secret": _KEYCLOAK_CLIENT_SECRET,
        },
        timeout=10,
        verify=False,
    )
    response.raise_for_status()
    body = response.json()
    _service_token = body["access_token"]
    _service_token_exp = now + float(body.get("expires_in", 300))
    return _service_token


# FIXME: Synthetic pools / reports are written directly to MinIO for now. Long term these
# writes should go through the data-catalogue / model-catalogue web API (same base_url as reads).
class DataCatalogueClient:
    """Client for the SYNTHEMA data catalogue.

    Two responsibilities, kept behind a single seam so the MinIO writes can later be
    swapped for catalogue-API calls without touching callers:

    * Read the per-node dataset assignments for a use case (catalogue HTTP API).
    * Write synthetic pools and validation reports to the MinIO data plane.

    The MinIO client is created lazily, so read-only callers (e.g. ``load_data``) do not
    require a reachable object store.
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or DATA_CATALOGUE_URL).rstrip("/")
        self.bucket_name = os.getenv("MINIO_BUCKET", "synthema-catalogue")
        self._minio: Optional[Minio] = None

    # ------------------------------------------------------------------
    # Catalogue HTTP API (reads)
    # ------------------------------------------------------------------
    def get_use_case_datasets(self, use_case: str) -> Dict[str, List[str]]:
        """Return the dataset paths assigned to each node for a use case.

        Returns:
            ``{"NODE1": ["path1.csv", "path2.csv"], "NODE2": ["path3.csv"]}``
        """
        response = requests.get(f"{self.base_url}/usecases", timeout=10)
        response.raise_for_status()
        data = response.json()

        for uc in data.get("use_cases", []):
            if uc.get("use_case") == use_case:
                return uc.get("datasets", {})

        raise ValueError(f"Use case {use_case} not found in catalogue")

    # ------------------------------------------------------------------
    # MinIO data plane (writes)
    # ------------------------------------------------------------------
    def _client(self) -> Minio:
        """Lazily build the MinIO client and guarantee the target bucket exists."""
        if self._minio is None:
            endpoint = os.getenv("MINIO_ENDPOINT", self.base_url)
            endpoint = endpoint.replace("http://", "").replace("https://", "")
            self._minio = Minio(
                endpoint=endpoint,
                access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
                secret_key=os.getenv("MINIO_SECRET_KEY", "adminpassword"),
                secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            )
            self._ensure_bucket()
        return self._minio

    def _ensure_bucket(self) -> None:
        try:
            if not self._minio.bucket_exists(self.bucket_name):
                self._minio.make_bucket(self.bucket_name)
                logger.info("Created MinIO bucket", bucket=self.bucket_name)
        except S3Error as e:
            logger.error("MinIO bucket validation failed", error=str(e))
            raise

    def upload_synthetic_pool(self, experiment_id: str, df: pd.DataFrame) -> str:
        """Stream the synthetic Big Pool DataFrame to MinIO, avoiding disk I/O."""
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        object_name = f"models/{experiment_id}/synthetic_big_pool.csv"

        self._client().put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=io.BytesIO(csv_bytes),
            length=len(csv_bytes),
            content_type="text/csv",
        )
        logger.info("Uploaded synthetic big pool to MinIO", object_name=object_name)
        return object_name

    def upload_html_report(self, experiment_id: str, node_id: str, report_type: str, html_content: str) -> str:
        """Upload an isolated utility/privacy validation HTML report to MinIO."""
        html_bytes = html_content.encode("utf-8")
        object_name = f"models/{experiment_id}/reports/node_{node_id}_{report_type}.html"

        self._client().put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=io.BytesIO(html_bytes),
            length=len(html_bytes),
            content_type="text/html",
        )
        logger.info("Uploaded validation report to MinIO", object_name=object_name, report_type=report_type)
        return object_name

    # ------------------------------------------------------------------
    # Catalogue HTTP API (writes) — register the pool for expert-committee review
    # ------------------------------------------------------------------
    def register_synthetic_pool(
        self,
        use_case: str,
        sdg_model_name: str,
        node_name: str,
        n_samples: int,
        pool_object: str,
        report_objects: Dict[str, Dict[str, str]],
    ) -> dict:
        """Register the generated Big Pool in the catalogue so it appears in the
        dashboard's expert-committee review queue (status ``pending_approval``).

        ``pool_object`` and the values in ``report_objects`` are MinIO object names
        (as returned by ``upload_synthetic_pool`` / ``upload_html_report``); we store
        them as ``s3://<bucket>/<object>`` URIs. Making those browser-openable is a
        follow-up (a public MinIO URL or a catalogue report-download endpoint).
        """

        def _s3(obj: str) -> str:
            return f"s3://{self.bucket_name}/{obj}"

        validation_reports: Dict[str, str] = {}
        for nid, reports in report_objects.items():
            for report_type, obj in reports.items():
                if obj:
                    validation_reports[f"node_{nid}_{report_type}_uri"] = _s3(obj)

        pool_metadata = {
            "node_name": node_name,
            "n_samples": int(n_samples),
            "s3_uris": {"pool_uri": _s3(pool_object)},
            "local_paths": {},
            "validation_reports": validation_reports,
        }

        headers = {}
        token = _get_catalogue_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.post(
            f"{self.base_url}/synthetic-pools",
            params={"use_case": use_case, "sdg_model_name": sdg_model_name},
            json=pool_metadata,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "Registered synthetic pool for expert-committee review",
            pool_id=result.get("pool_id"),
            use_case=use_case,
        )
        return result
