import os

MLFLOW_URL = os.environ["MLFLOW_URL"]

RABBIT_USERNAME = os.environ["RABBIT_USERNAME"]
RABBIT_PASSWORD = os.environ["RABBIT_PASSWORD"]
RABBIT_HOST = os.environ["RABBIT_HOST"]
RABBIT_PORT = os.environ["RABBIT_PORT"]

global global_vars
global_vars = {
    "aggregator": None,
    "model": None,
    "model_meta": None,
    "use_case": None,
}
