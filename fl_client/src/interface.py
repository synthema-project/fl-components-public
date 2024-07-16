import mlflow
from flwr.common import Message, ParametersRecord
from mlflow import MlflowClient
from fl_client.src.stages import (
    get_model_parameters,
    load_data,
    load_model,
    prepare_data,
    set_model_parameters,
    set_run_config,
    train_model,
    upload_model,
)
from common.utils.src.flower_utils.responses import (
    create_parameters_response,
    create_success_response,
    create_train_response,
    create_upload_model_response,
)


def load_data_if(msg: Message, global_vars: dict) -> Message:
    use_case = msg.content.configs_records["config"]["use_case"]
    if not isinstance(use_case, str):
        raise TypeError(f"Use case must be a string, got {type(use_case)}")
    load_data(use_case, global_vars)
    return create_success_response(msg, "loaded data successfully")


def prepare_data_if(msg: Message, global_vars: dict) -> Message:
    prepare_data(global_vars)
    return create_success_response(msg, "prepared data successfully")


def load_model_if(
    msg: Message, mlflow_client: MlflowClient, global_vars: dict
) -> Message:
    model_name = msg.content.configs_records["config"]["model_name"]
    model_version = msg.content.configs_records["config"]["model_version"]
    if not isinstance(model_name, str) or not isinstance(model_version, int):
        raise TypeError(
            f"Model name must be a string and model version must be an integer, got {type(model_name)} and {type(model_version)}"
        )
    load_model(model_name, model_version, mlflow_client, global_vars)
    return create_success_response(msg, "loaded model successfully")


def upload_model_if(
    msg: Message, mlflow_module: mlflow, mlflow_client: MlflowClient, global_vars: dict
) -> Message:
    latest_parameters: ParametersRecord = msg.content.parameters_records["parameters"]
    model_info = upload_model(
        latest_parameters, mlflow_module, mlflow_client, global_vars
    )
    return create_upload_model_response(msg, model_info.model_uuid, model_info.run_id)


def train_model_if(msg: Message, mlflow_module: mlflow, global_vars: dict) -> Message:
    latest_parameters: ParametersRecord = msg.content.parameters_records["parameters"]
    current_global_iter = msg.content.configs_records["config"]["current_global_iter"]
    if not isinstance(current_global_iter, int):
        raise TypeError(
            f"Current global iteration must be an integer, got {type(current_global_iter)}"
        )
    set_model_parameters(latest_parameters, global_vars)
    metrics = train_model(global_vars, current_global_iter, mlflow_module)
    assert (
        global_vars["mlflow_experiment_id"] is not None
        and global_vars["mlflow_run_id"] is not None
    )
    latest_parameters = get_model_parameters(global_vars)
    return create_train_response(msg, latest_parameters, metrics)


def get_parameters_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters = get_model_parameters(global_vars)
    return create_parameters_response(msg, latest_parameters)


def set_parameters_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters: ParametersRecord = msg.content.parameters_records["parameters"]
    set_model_parameters(latest_parameters, global_vars)
    return create_success_response(msg, "set parameters successfully")


def set_run_config_if(
    msg: Message, mlflow_client: MlflowClient, global_vars: dict
) -> Message:
    experiment_id = msg.content.configs_records["config"]["experiment_id"]
    run_id = msg.content.configs_records["config"]["run_id"]
    if not isinstance(experiment_id, str) or not isinstance(run_id, str):
        raise TypeError(
            f"Experiment ID and run ID must be str, got {type(experiment_id)} and {type(run_id)}"
        )
    set_run_config(mlflow_client, experiment_id, run_id, global_vars)
    return create_success_response(msg, "set run config successfully")
