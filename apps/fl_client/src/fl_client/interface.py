from collections import OrderedDict
from fl_models.iris.utils import Utils
from flwr.common import Message
from flwr.common.differential_privacy import (
    compute_clip_model_update,
)


from fl_client import stages
from fl_client.utils import responses_utils

from interfaces import recordset


def set_dp_config_if(msg: Message, global_vars: dict) -> Message:
    dp_params = recordset.read_set_dp_cfg_recordset(msg.content)
    stages.set_dp_config(dp_params,global_vars)
    return responses_utils.create_success_response(msg, "set dp config successfully")

def load_data_if(msg: Message, global_vars: dict) -> Message:
    use_case = recordset.read_load_data_recordset(msg.content)
    stages.load_data(use_case, global_vars)
    return responses_utils.create_success_response(msg, "loaded data successfully")

def prepare_data_if(msg: Message, global_vars: dict) -> Message:
    stages.prepare_data(global_vars)
    return responses_utils.create_success_response(msg, "prepared data successfully")


def load_model_if(msg: Message, global_vars: dict) -> Message:
    stages.load_model(global_vars)
    return responses_utils.create_success_response(msg, "loaded model successfully")


def upload_model_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters = recordset.read_upload_model_recordset(msg.content)
    model_info = stages.upload_model(latest_parameters, global_vars)
    return responses_utils.create_upload_model_response(
        msg, model_info["model_id"], model_info["run_id"]
    )


def train_model_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters, current_global_iter = recordset.read_train_model_recordset(
        msg.content
    )
    stages.set_model_parameters(latest_parameters, global_vars)
    metrics = stages.train_model(global_vars, current_global_iter)
    latest_parameters = stages.get_model_parameters(global_vars)
    return responses_utils.create_train_response(msg, latest_parameters, metrics)

def train_model_fixed_clipping_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters, current_global_iter = recordset.read_train_model_recordset(
        msg.content
    )
    # Convert parameters from server to client in ndarrays 
    server_to_client_params_dict = Utils.parameters_to_dict(latest_parameters)
    server_to_client_params = list(server_to_client_params_dict.values())

    stages.set_model_parameters(latest_parameters, global_vars)
    metrics = stages.train_model(global_vars, current_global_iter)
    latest_parameters = stages.get_model_parameters(global_vars)

    # Convert parameters from client to server in ndarrays
    client_to_server_params_dict = Utils.parameters_to_dict(latest_parameters)
    client_to_server_params = list(client_to_server_params_dict.values())

    # Clip the client update
    compute_clip_model_update(
        client_to_server_params,
        server_to_client_params,
        global_vars["dp_config"]["clipping_norm"],
    )

    clipped_params = OrderedDict()
    keys = client_to_server_params_dict.keys()

    for key, value in zip(keys,client_to_server_params):
        clipped_params[key] = value

    clipped_parameters=Utils.dict_to_parameter_record(clipped_params)
   

    return responses_utils.create_train_response(msg, clipped_parameters, metrics)

def get_parameters_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters = stages.get_model_parameters(global_vars)
    return responses_utils.create_parameters_response(msg, latest_parameters)


def set_parameters_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters = recordset.read_set_parameters_recordset(msg.content)
    stages.set_model_parameters(latest_parameters, global_vars)
    return responses_utils.create_success_response(msg, "set parameters successfully")


def set_run_config_if(msg: Message, global_vars: dict) -> Message:
    content = recordset.read_set_run_cfg_recordset(msg.content)
    stages.set_run_config(
        content["experiment_id"],
        content["parent_run_id"],
        global_vars["node_name"],
        content["model_name"],
        content["model_version"],
        global_vars["data_path"],
    )
    return responses_utils.create_success_response(msg, "set run config successfully")


def clean_config_if(msg: Message) -> Message:
    stages.clean_current_config()
    return responses_utils.create_success_response(msg, "cleaned config successfully")


def filter_clients_if(msg: Message, global_vars: dict) -> Message:
    use_case = recordset.read_filter_clients_recordset(msg.content)
    if use_case == global_vars["use_case"]:
        return responses_utils.create_participate_response(msg, True)
    return responses_utils.create_participate_response(msg, False)
