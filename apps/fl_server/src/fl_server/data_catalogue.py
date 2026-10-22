from interfaces.catalogue_auth import catalogue_headers
from interfaces.validation_policy import required_reports
import requests
import json
from typing import Any, Dict, List, Optional, Tuple


class DataCatalogueClient:
    def __init__(self, base_url: str, token: str = ""):
        self.base_url = base_url
        self.token = token

    @property
    def headers(self) -> dict[str, str]:
        return catalogue_headers(self.token)

    def get_use_case_datasets(self, use_case: str) -> Dict[str, Any]:
        """
        Returns:
        {
            "NODE1": ["path1.csv", "path2.csv"],
            "NODE2": ["path3.csv"]
        }
        """
        # The typed per-file records are the source of truth for new uploads.
        # Fall back to /usecases for catalogues populated before dataset_role
        # was introduced.
        records_response = requests.get(f"{self.base_url}/metadata", headers=self.headers)
        records_response.raise_for_status()
        records = [r for r in records_response.json().get("datasets", [])
                   if r.get("use_case") == use_case]
        typed: Dict[str, Dict[str, str]] = {}
        self.selected_documents = {}
        groups = {}
        for record in records:
            if record.get("collection_id"):
                key = (record["node"], record["collection_id"])
                groups.setdefault(key, []).append(record)
        # Never assemble a dataset from partitions of different uploads.
        latest = {}
        for (node, collection), rows in groups.items():
            roles = {row.get("dataset_role"): row for row in rows}
            if not {"train", "test", "schema"}.issubset(roles):
                continue  # Incomplete uploads are not selectable.
            stamp = max(str(roles[role].get("timestamp", "")) for role in ("train", "test", "schema"))
            if node not in latest or stamp > latest[node][0]:
                latest[node] = (stamp, roles)
        if groups:
            for node, (_, roles) in latest.items():
                typed[node] = {role: roles[role]["path"] for role in ("train", "test", "schema")}
                versions = {roles[role].get("configuration_version") for role in ("train", "test", "schema")}
                if len(versions) != 1 or None in versions:
                    raise ValueError("Dataset partitions do not share a configuration version")
                row = roles["train"]
                self.selected_documents[node] = (
                    json.dumps(row["dataset_meta"]), json.dumps(row["evaluation_configuration"]))
            return typed
        for record in records:
            role = record.get("dataset_role")
            if role in {"train", "test", "schema"}:
                typed.setdefault(record["node"], {})[role] = record["path"]
        if typed:
            return typed

        response = requests.get(f"{self.base_url}/usecases/{use_case}", headers=self.headers)

        response.raise_for_status()
        data = response.json()

        if data.get("use_case") != use_case:
            raise ValueError(f"Use case {use_case} not found in catalogue")
        return data.get("datasets", {})

    def get_use_case_metadata(self, use_case: str) -> Tuple[Optional[str], Optional[str]]:
        """Return the validated metadata documents associated with a use case.

        The catalogue response may omit either document while deployments are
        being migrated. Empty values are returned as ``None`` so callers can
        retain the legacy execution path until both documents are available.
        """
        response = requests.get(f"{self.base_url}/metadata", headers=self.headers)
        response.raise_for_status()
        data = response.json()

        for dataset in data.get("datasets", []):
            if dataset.get("use_case") != use_case:
                continue
            metadata = dataset.get("dataset_meta") or dataset.get("dataset_metadata")
            if metadata is not None:
                # The current catalogue stores dataset metadata only. The
                # evaluation configuration is supplied by the registered
                # model, so leave it absent here and let the caller use that
                # model-owned document when available.
                evaluation = dataset.get("evaluation_configuration")
                return json.dumps(metadata), json.dumps(evaluation) if evaluation is not None else None
        return None, None

    def register_synthetic_pool(
        self,
        *,
        pool_id: str,
        use_case: str,
        model_name: str,
        node_name: str,
        n_samples: int,
        object_uri: str,
        metadata_uri: str | None = None,
        evaluation_uri: str | None = None,
        token: str,
        validation_nodes: list[str] | None = None,
        dataset_snapshot: dict | None = None,
        task_id: int | None = None,
    ) -> dict[str, Any]:
        """Register a generated object in the catalogue approval queue."""
        response = requests.post(
            f"{self.base_url}/synthetic-pools",
            params={"use_case": use_case, "sdg_model_name": model_name},
            json={
                "pool_id": pool_id,
                "validation_phase": "awaiting_reports",
                "validation_nodes": validation_nodes or [],
                "required_reports": required_reports(),
                "dataset_snapshot": dataset_snapshot,
                "task_id": task_id,
                "node_name": node_name,
                "n_samples": n_samples,
                "s3_uris": {
                    "synthetic_data": object_uri,
                    **({"metadata": metadata_uri} if metadata_uri else {}),
                    **({"evaluation_configuration": evaluation_uri} if evaluation_uri else {}),
                },
            },
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def register_pool_validation(self, pool_id: str, node_name: str,
                                 metrics: dict, report_uri: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/synthetic-pools/{pool_id}/validation",
            json={"node_name": node_name, "metrics": metrics, "report_uri": report_uri},
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()
