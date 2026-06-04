import requests
from typing import Dict, List


class DataCatalogueClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_use_case_datasets(self, use_case: str) -> Dict[str, List[str]]:
        """
        Returns:
        {
            "NODE1": ["path1.csv", "path2.csv"],
            "NODE2": ["path3.csv"]
        }
        """
        response = requests.get(f"{self.base_url}/catalogue")

        response.raise_for_status()
        data = response.json()

        for uc in data.get("use_cases", []):
            if uc["use_case"] == use_case:
                return uc.get("datasets", {})

        raise ValueError(f"Use case {use_case} not found in catalogue")

    def get_node_info(self, node_name: str) -> Dict:
        """
        Fetch node capability/metadata from the catalogue.
        Returns a dict with keys like score, cpu_cores, cpu, capacity, etc.
        """
        try:
            response = requests.get(f"{self.base_url}/nodes/{node_name}", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception:
            # If endpoint not available, return empty dict (will cause fallback to random)
            return {}

    def register_synthetic_pool(self, use_case: str, sdg_model_name: str,
                                pool_metadata: Dict) -> Dict:
        """
        Register a newly generated synthetic data pool to the catalogue.
        Stores metadata: s3_uris, validation_reports, node_name, status, etc.
        Returns the registration record with a pool_id.
        """
        payload = {
            "use_case": use_case,
            "sdg_model_name": sdg_model_name,
            "pool_metadata": pool_metadata,
            "status": "pending_approval"
        }
        try:
            response = requests.post(f"{self.base_url}/synthetic-pools", json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to register synthetic pool: {e}")

    def approve_synthetic_pool(self, pool_id: str) -> Dict:
        """
        Approve a synthetic data pool (expert committee decision).
        Marks it as published and available for use.
        """
        try:
            response = requests.post(f"{self.base_url}/synthetic-pools/{pool_id}/approve", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to approve pool {pool_id}: {e}")

    def reject_synthetic_pool(self, pool_id: str, reason: str = "") -> Dict:
        """
        Reject a synthetic data pool.
        """
        try:
            response = requests.post(f"{self.base_url}/synthetic-pools/{pool_id}/reject",
                                   json={"reason": reason}, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to reject pool {pool_id}: {e}")

    def get_synthetic_pool(self, pool_id: str) -> Dict:
        """
        Fetch synthetic pool record including status, validation reports, s3_uris, etc.
        """
        try:
            response = requests.get(f"{self.base_url}/synthetic-pools/{pool_id}", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch pool {pool_id}: {e}")

