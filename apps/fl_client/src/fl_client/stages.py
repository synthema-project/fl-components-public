import pandas as pd
from flwr.common import MetricsRecord, ParametersRecord
from mlflow.models.model import ModelInfo

from fl_client.config import DATA_PATH

from interfaces import mlflow_client


def set_run_config(
    experiment_id: str,
    parent_run_id: str,
    node_name: str,
    model_name: str,
    model_version: int,
    data_path: str,
) -> None:
    child_run_id = mlflow_client.create_child_run(
        experiment_id, parent_run_id, node_name
    )
    mlflow_client.set_current_config(
        experiment_id, parent_run_id, child_run_id, model_name, model_version
    )
    mlflow_client.set_dataset_signature(node_name, data_path, child_run_id)


def load_data(use_case: str, global_vars: dict) -> None:
    if use_case != "iris":
        raise NotImplementedError(f"Unknown use case: {use_case}")
    global_vars["use_case"] = use_case
    global_vars["data"] = pd.read_csv(DATA_PATH)


def prepare_data(global_vars: dict) -> None:
    local_learner = global_vars["local_learner"]
    data = global_vars["data"]
    local_learner.prepare_data(data=data)


def load_model(
    global_vars: dict,
) -> None:
    mlf_model = mlflow_client.load_model()
    global_vars["model"] = mlf_model
    global_vars["local_learner"] = (
        mlf_model.unwrap_python_model().create_local_learner()
    )


def train_model(
    global_vars: dict,
    current_global_iter: int,
) -> MetricsRecord:
    local_learner = global_vars["local_learner"]
    metrics = local_learner.train()
    if not isinstance(metrics, MetricsRecord):
        raise TypeError(f"Unexpected metrics type: {type(metrics)}")
    mlflow_client.log_metrics(metrics, current_global_iter)
    return metrics


def evaluate_model(global_vars: dict) -> MetricsRecord:
    local_learner = global_vars["local_learner"]
    metrics = local_learner.evaluate()
    if not isinstance(metrics, MetricsRecord):
        raise TypeError(f"Expected MetricsRecord, got {type(metrics)}")
    return metrics


def get_model_parameters(
    global_vars: dict,
) -> ParametersRecord:
    local_learner = global_vars["local_learner"]
    parameters = local_learner.get_parameters()
    if not isinstance(parameters, ParametersRecord):
        raise TypeError(f"Expected ParametersRecord, got {type(parameters)}")
    return parameters


def set_model_parameters(parameters: ParametersRecord, global_vars: dict) -> None:
    local_learner = global_vars["local_learner"]
    local_learner.set_parameters(parameters)


def upload_model(
    latest_parameters: ParametersRecord,
    global_vars: dict,
) -> ModelInfo:
    local_learner = global_vars["local_learner"]
    set_model_parameters(latest_parameters, global_vars)
    model_info = mlflow_client.upload_final_state(local_learner)
    return model_info


def clean_current_config() -> None:
    mlflow_client.clean_current_config()
