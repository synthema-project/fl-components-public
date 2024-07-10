import os

MLFLOW_URL = os.environ["MLFLOW_URL"]

global global_vars
global_vars = {
    "aggregator": None,
    "model": None,
    "model_meta": None,
    "use_case": None,
    "mlflow_experiment_id": None,
    "mlflow_run_id": None,
}
