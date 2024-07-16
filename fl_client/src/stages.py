import mlflow
import pandas as pd
from flwr.common import MetricsRecord, ParametersRecord
from mlflow.models.model import ModelInfo
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from fl_client.src.config import DATA_PATH
from common.utils.src.mlflow_utils import (
    load_mlflow_model,
    register_model_metadata,
    upload_final_state,
)


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
    model_name: str,
    model_version: int,
    mlflow_client: mlflow.MlflowClient,
    global_vars: dict,
) -> None:
    mlf_model, mlf_model_meta = load_mlflow_model(
        mlflow_client, model_name=model_name, model_version=model_version
    )
    global_vars["model"] = mlf_model
    global_vars["model_meta"] = mlf_model_meta
    global_vars["local_learner"] = (
        mlf_model.unwrap_python_model().create_local_learner()
    )


def set_run_config(
    mlflow_client: mlflow.MlflowClient,
    experiment_id: str,
    run_id: str,
    global_vars: dict,
) -> None:
    global_vars["mlflow_experiment_id"] = experiment_id
    run = mlflow_client.create_run(
        experiment_id=experiment_id,
        run_name=global_vars["node_name"],
        tags={MLFLOW_PARENT_RUN_ID: run_id},
    )
    from mlflow.data.meta_dataset import MetaDataset
    from mlflow.data.http_dataset_source import HTTPDatasetSource

    source = HTTPDatasetSource(f"{global_vars['node_name']}://{DATA_PATH}")
    # source._get_source_type = lambda: "local"
    dataset = MetaDataset(source)
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_input(dataset, "training")
    global_vars["mlflow_run_id"] = run.info.run_id


def train_model(
    global_vars: dict,
    current_global_iter: int,
    mlflow_module: mlflow,
) -> MetricsRecord:
    local_learner = global_vars["local_learner"]
    metrics = local_learner.train()
    if not isinstance(metrics, MetricsRecord):
        raise TypeError(f"Unexpected metrics type: {type(metrics)}")
    with mlflow_module.start_run(run_id=global_vars["mlflow_run_id"]):
        mlflow_module.log_metrics(metrics, step=current_global_iter)
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
    mlflow_module: mlflow,
    mlflow_client: mlflow.MlflowClient,
    global_vars: dict,
) -> ModelInfo:
    # Unpack global_vars
    local_learner = global_vars["local_learner"]
    set_model_parameters(latest_parameters, global_vars)
    model_meta = global_vars["model_meta"]
    mlflow_run_id = global_vars["mlflow_run_id"]

    parent_run_id = mlflow_client.get_parent_run(mlflow_run_id).info.run_id

    model_info: ModelInfo = upload_final_state(
        mlflow_module, local_learner, model_meta, parent_run_id
    )
    register_model_metadata(mlflow_client, model_meta)
    return model_info
