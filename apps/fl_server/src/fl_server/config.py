import os

MLFLOW_URL: str = os.environ["MLFLOW_URL"]

RABBIT_USERNAME: str = os.environ["RABBIT_USERNAME"]
RABBIT_PASSWORD: str = os.environ["RABBIT_PASSWORD"]
RABBIT_HOST: str = os.environ["RABBIT_HOST"]
RABBIT_PORT: str = os.getenv("RABBIT_PORT", "5672")

DATA_CATALOGUE_URL: str = os.getenv("DATA_CATALOGUE_URL", "http://data-catalogue:8003")

P2P_PORT: int = int(os.getenv("P2P_PORT", "50051"))

global global_vars
global_vars = {
    "aggregator": None,
    "model": None,
    "model_meta": None,
    "use_case": None,
    "dp_config": None,
    "number_of_clients":None
}
