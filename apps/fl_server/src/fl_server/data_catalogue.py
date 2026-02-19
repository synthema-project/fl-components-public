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
