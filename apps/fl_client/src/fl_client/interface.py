import threading
from collections import OrderedDict
from flwr.common import Message
import structlog
from fl_models.iris.utils import Utils
from flwr.common.differential_privacy import (
    compute_clip_model_update,
)
from interfaces import recordset
from fl_client import stages
from fl_client.utils import responses_utils
from fl_client.utils import peer_discovery
from fl_client.utils.parameters_utils import serialize_parameters, deserialize_parameters, split_model_parameters_to_shares, numpy_dict_to_torch_state_dict
from fl_client.utils.responses_utils import create_error_response

logger = structlog.get_logger()

def set_dp_config_if(msg: Message, global_vars: dict) -> Message:
    dp_params = recordset.read_set_dp_cfg_recordset(msg.content)
    stages.set_dp_config(dp_params,global_vars)
    return responses_utils.create_success_response(msg, "set dp config successfully")

def load_data_if(msg: Message, global_vars: dict) -> Message:
    use_case, dataset_paths, metadata_json, evaluation_config_json = recordset.read_load_data_recordset(msg.content)

    stages.load_data(use_case, dataset_paths, global_vars, metadata_json, evaluation_config_json)

    return responses_utils.create_success_response(
        msg, "loaded data successfully"
    )


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


def generate_synthetic_data_if(msg: Message, global_vars: dict) -> Message:
    n_samples = recordset.read_generate_synthetic_data_recordset(msg.content)
    result = stages.generate_synthetic_data(
        n_samples, global_vars, str(msg.content.configs_records["config"]["pool_id"])
    )
    return responses_utils.create_synthetic_data_response(
        msg,
        result["object_uri"],
        global_vars["node_name"],
        result["metadata_uri"],
        result["evaluation_uri"],
        result.get("utility_reporting_enabled", False),
    )


def validate_synthetic_pool_if(msg: Message, global_vars: dict) -> Message:
    pool_id, object_uri, use_case = recordset.read_validate_synthetic_pool_recordset(msg.content)
    result = stages.validate_synthetic_pool(pool_id, object_uri, use_case, global_vars)
    return responses_utils.create_validation_response(msg, result)

def set_smpc_enabled_if(msg: Message, global_vars: dict) -> Message:
    smpc_enabled = recordset.read_set_smpc_enabled_recordset(msg.content)
    stages.set_SMPC_enabled(smpc_enabled, global_vars)
    return responses_utils.create_success_response(msg, "set smpc enabled successfully")

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

def train_model_SMPC_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters, current_global_iter = recordset.read_train_model_recordset(
        msg.content
    )
    stages.set_model_parameters(latest_parameters, global_vars)
    metrics = stages.train_model(global_vars, current_global_iter)
    latest_parameters = stages.get_model_parameters(global_vars)

    serialized_params = serialize_parameters(latest_parameters)
    deserialized_params = deserialize_parameters(serialized_params)

    all_node_names = list(global_vars["discovered_peers_ids"])
    all_node_names.append(global_vars["node_name"])
    local_shares, remote_shares = split_model_parameters_to_shares(deserialized_params, all_node_names, global_vars["node_name"])

    logger.info(deserialized_params)
    print("-- Sending shares to peers --")
    stages.send_shares_to_peers(remote_shares, global_vars["peer_stubs"], global_vars["node_name"])
    latest_parameters = stages.reconstruct_shares(local_shares, global_vars['smpc_client'])

    latest_parameters = numpy_dict_to_torch_state_dict(latest_parameters)
    parameter_record = Utils.pytorch_to_parameter_record(latest_parameters)
    return responses_utils.create_train_response(msg, parameter_record, metrics)

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
    global_vars["model_name"] = content["model_name"]
    global_vars["model_version"] = content["model_version"]
    return responses_utils.create_success_response(msg, "set run config successfully")

def clean_config_if(msg: Message) -> Message:
    stages.clean_current_config(msg.content.configs_records["config"].get("status", "FINISHED"))
    return responses_utils.create_success_response(msg, "cleaned config successfully")

def filter_clients_if(msg: Message, global_vars: dict) -> Message:
    use_case = recordset.read_filter_clients_recordset(msg.content)
    # Each deployment is dedicated to one use case.  When FL_USE_CASE is
    # configured, reject tasks for another network; when it is not configured,
    # accept the task instead of applying the obsolete iris default.
    configured_use_case = global_vars.get("use_case")
    if not configured_use_case or use_case == configured_use_case:
        return responses_utils.create_participate_response(msg, True, global_vars["node_name"])
    return responses_utils.create_participate_response(msg, False, global_vars["node_name"])

def get_peer_discovery_if(msg: Message, global_vars: dict) -> Message:
    config = recordset.read_discover_peers_recordset(msg.content)
    logger.info(f"DISCOVER CONFIG → {config}")

    discovery_service = peer_discovery.PeerDiscovery(
        client_id=global_vars["node_name"],
        grpc_port=config['client_port'],
        min_peers=config['min_peers'],
        max_discovery_time=config['max_discovery_time'],)

    # Start discovery in a separate thread
    discovery_thread = threading.Thread(
        target=discovery_service.start_discovery)
    discovery_thread.start()
    discovery_thread.join()
    discovered_peers = discovery_service.get_peer_addresses()
    discovered_peers_ids = discovery_service.get_peer_ids()
    logger.info(f"Discovered {len(discovered_peers)} peers: {discovered_peers}")
    stages.set_discovered_peers(discovered_peers, discovered_peers_ids, global_vars)
    return responses_utils.create_discovery_response(msg, discovered_peers)

def connect_peers_if(msg: Message, global_vars: dict) -> Message:
    discovered_peers = global_vars["discovered_peers"]
    logger.info("-- CLEINT DISCOVERED PEERS --")
    logger.info(discovered_peers)
    if discovered_peers is None or len(discovered_peers) == 0:
        return responses_utils.create_error_response(msg, "No peers discovered")
    global_vars["peer_stubs"] = stages.connect_peers(discovered_peers, global_vars)
    return responses_utils.create_connect_peers_response(msg, discovered_peers)

def start_P2P_server_if(msg: Message, global_vars: dict) -> Message:
    config = recordset.read_start_P2P_server_recordset(msg.content)
    stages.start_P2P_server(
        global_vars['node_name'], config['port'], global_vars
    )

    return responses_utils.create_start_p2p_server_response(msg, config['port'])

def generate_synthetic_if(msg: Message, global_vars: dict) -> Message:
    try:
        params = recordset.read_generate_synthetic_recordset(msg.content)
    except Exception as e:
        return create_error_response(msg, f"Invalid generate_synthetic recordset: {e}")

    try:
        # Optional legacy handler must not prevent ClientApp startup.
        from fl_client.utils.responses_utils import create_generate_synthetic_response

        result = stages.generate_synthetic_and_upload(
            global_vars,
            params["n_samples"],
            params["sdg_model_name"],
            params.get("dataset_meta_json", ""),
            params.get("upload_to_s3", True),
            params.get("s3_bucket", "synthetic-data"),
            params.get("create_s3_bucket_if_missing", False),
            params.get("real_data_partition", "test"),
        )
        return create_generate_synthetic_response(msg, result)
    except Exception as e:
        return create_error_response(msg, f"Synthetic generation failed: {e}")
