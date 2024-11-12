import os

MLFLOW_URL = os.getenv("MLFLOW_URL")

RABBIT_USERNAME = os.getenv("RABBIT_USERNAME")
RABBIT_PASSWORD = os.getenv("RABBIT_PASSWORD")
RABBIT_HOST = os.getenv("RABBIT_HOST")
RABBIT_PORT = os.getenv("RABBIT_PORT", "5672")

global global_vars
global_vars = {
    "aggregator": None,
    "model": None,
    "model_meta": None,
    "use_case": None,
}
