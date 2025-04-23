from flwr.common import Message, parameters_to_ndarrays, Parameters
import threading
import structlog
from fl_client import stages
from fl_client.utils import responses_utils
from fl_client.utils import peer_discovery
from interfaces import recordset
from fl_client.utils.parameters_utils import serialize_parameters, deserialize_parameters, split_model_parameters_to_shares, numpy_dict_to_torch_state_dict
from fl_models.iris.utils import Utils

logger = structlog.get_logger()

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
    print("-----")
    print(metrics)
    print("-----")
    latest_parameters = stages.get_model_parameters(global_vars)

    serialized_params = serialize_parameters(latest_parameters)
    deserialized_params = deserialize_parameters(serialized_params)

    all_node_names = list(global_vars["discovered_peers_ids"])
    all_node_names.append(global_vars["node_name"])
    local_shares, remote_shares = split_model_parameters_to_shares(deserialized_params, all_node_names, global_vars["node_name"])

    logger.info(deserialized_params)
    print("-- SENDING SHARES TO PEERS --")
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
    return responses_utils.create_success_response(msg, "set run config successfully")


def clean_config_if(msg: Message) -> Message:
    stages.clean_current_config()
    return responses_utils.create_success_response(msg, "cleaned config successfully")


def filter_clients_if(msg: Message, global_vars: dict) -> Message:
    use_case = recordset.read_filter_clients_recordset(msg.content)
    if use_case == global_vars["use_case"]:
        return responses_utils.create_participate_response(msg, True)
    return responses_utils.create_participate_response(msg, False)


def get_peer_discovery_if(msg: Message, global_vars: dict) -> Message:
    config = recordset.read_discover_peers_recordset(msg.content)

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