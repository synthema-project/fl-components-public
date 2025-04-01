import os

MLFLOW_URL: str = os.environ["MLFLOW_URL"]

RABBIT_USERNAME: str = os.environ["RABBIT_USERNAME"]
RABBIT_PASSWORD: str = os.environ["RABBIT_PASSWORD"]
RABBIT_HOST: str = os.environ["RABBIT_HOST"]
RABBIT_PORT: str = os.getenv("RABBIT_PORT", "5672")

P2P_PORT: int = int(os.getenv("P2P_PORT", "50051"))

global global_vars
global_vars = {
    "aggregator": None,
    "model": None,
    "model_meta": None,
    "use_case": None,
}
