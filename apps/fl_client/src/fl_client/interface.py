from flwr.common import Message, ParametersRecord

from fl_client import stages

from fl_client.utils import responses_utils


def load_data_if(msg: Message, global_vars: dict) -> Message:
    use_case = msg.content.configs_records["config"]["use_case"]
    if not isinstance(use_case, str):
        raise TypeError(f"Use case must be a string, got {type(use_case)}")
    stages.load_data(use_case, global_vars)
    return responses_utils.create_success_response(msg, "loaded data successfully")


def prepare_data_if(msg: Message, global_vars: dict) -> Message:
    stages.prepare_data(global_vars)
    return responses_utils.create_success_response(msg, "prepared data successfully")


def load_model_if(msg: Message, global_vars: dict) -> Message:
    stages.load_model(global_vars)
    return responses_utils.create_success_response(msg, "loaded model successfully")


def upload_model_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters: ParametersRecord = msg.content.parameters_records["parameters"]
    model_info = stages.upload_model(latest_parameters, global_vars)
    # TODO dump model info
    return responses_utils.create_upload_model_response(
        msg, model_info["model_id"], model_info["run_id"]
    )


def train_model_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters: ParametersRecord = msg.content.parameters_records["parameters"]
    current_global_iter = msg.content.configs_records["config"]["current_global_iter"]
    if not isinstance(current_global_iter, int):
        raise TypeError(
            f"Current global iteration must be an integer, got {type(current_global_iter)}"
        )
    stages.set_model_parameters(latest_parameters, global_vars)
    metrics = stages.train_model(global_vars, current_global_iter)
    latest_parameters = stages.get_model_parameters(global_vars)
    return responses_utils.create_train_response(msg, latest_parameters, metrics)


def get_parameters_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters = stages.get_model_parameters(global_vars)
    return responses_utils.create_parameters_response(msg, latest_parameters)


def set_parameters_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters: ParametersRecord = msg.content.parameters_records["parameters"]
    stages.set_model_parameters(latest_parameters, global_vars)
    return responses_utils.create_success_response(msg, "set parameters successfully")


def set_run_config_if(msg: Message, global_vars: dict) -> Message:
    experiment_id = msg.content.configs_records["config"]["experiment_id"]
    parent_run_id = msg.content.configs_records["config"]["run_id"]
    model_name = msg.content.configs_records["config"]["model_name"]
    model_version = msg.content.configs_records["config"]["model_version"]
    if (
        not isinstance(experiment_id, str)
        or not isinstance(parent_run_id, str)
        or not isinstance(model_name, str)
        or not isinstance(model_version, int)
    ):
        raise TypeError(
            f"Experiment ID and run ID must be str, got {type(experiment_id)} and {type(parent_run_id)}"
        )
    stages.set_run_config(
        experiment_id,
        parent_run_id,
        global_vars["node_name"],
        model_name,
        model_version,
        global_vars["data_path"],
    )
    return responses_utils.create_success_response(msg, "set run config successfully")


def clean_config_if(msg: Message) -> Message:
    stages.clean_current_config()
    return responses_utils.create_success_response(msg, "cleaned config successfully")


def filter_clients_if(msg: Message, global_vars: dict) -> Message:
    use_case = msg.content.configs_records["config"]["use_case"]
    if not isinstance(use_case, str):
        raise TypeError(f"Use case must be a string, got {type(use_case)}")
    if use_case == global_vars["use_case"]:
        return responses_utils.create_participate_response(msg, True)
    return responses_utils.create_participate_response(msg, False)
