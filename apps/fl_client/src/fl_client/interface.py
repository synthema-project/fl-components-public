import threading
from collections import OrderedDict
from flwr.common import Message, parameters_to_ndarrays, Parameters
import structlog
from fl_models.iris.utils import Utils
from flwr.common import Message
from flwr.common.differential_privacy import (
    compute_clip_model_update,
)
from interfaces import recordset
from model_store_interface import DatasetMeta, UseCase

from fl_client import stages
from fl_client.utils import peer_discovery
from fl_client.utils import responses_utils
from fl_client.utils.parameters_utils import serialize_parameters, deserialize_parameters, \
    split_model_parameters_to_shares, numpy_dict_to_torch_state_dict
from fl_client.utils.parameters_utils import serialize_parameters, deserialize_parameters, split_model_parameters_to_shares, numpy_dict_to_torch_state_dict
from fl_client.utils.responses_utils import create_error_response, create_generate_synthetic_response


logger = structlog.get_logger()


def set_dp_config_if(msg: Message, global_vars: dict) -> Message:
    dp_params = recordset.read_set_dp_cfg_recordset(msg.content)
    stages.set_dp_config(dp_params, global_vars)
    return responses_utils.create_success_response(msg, "set dp config successfully")


def load_data_if(msg: Message, global_vars: dict) -> Message:
    use_case, node_dataset_map = recordset.read_load_data_recordset(msg.content)

    node_name = global_vars["node_name"]
    dataset_paths = node_dataset_map.get(node_name, [])

    global_vars["use_case"] = use_case
    stages.load_data(use_case, dataset_paths, global_vars)

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

    for key, value in zip(keys, client_to_server_params):
        clipped_params[key] = value

    clipped_parameters = Utils.dict_to_parameter_record(clipped_params)
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
    local_shares, remote_shares = split_model_parameters_to_shares(deserialized_params, all_node_names,
                                                                   global_vars["node_name"])

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
    dataset_meta_json = content.get("dataset_meta_json")

    if dataset_meta_json:
        try:
            # Deserialise the string into the type-safe Python >=3.11 object
            dataset_meta = DatasetMeta.model_validate_json(dataset_meta_json)
            if isinstance(dataset_meta.use_case, str):
                dataset_meta = dataset_meta.model_copy(
                    update={"use_case": UseCase(dataset_meta.use_case)}
                )
            global_vars["dataset_meta"] = dataset_meta

            # Safely log the enum value to guarantee type alignment
            logger.info(
                f"Successfully hydrated DCAT-AP schema for Use Case: {dataset_meta.use_case.value}"
            )
        except Exception as e:
            logger.error(f"Catastrophic failure hydrating Pydantic DatasetMeta: {e}")
            global_vars["dataset_meta"] = None
    else:
        logger.warning(
            "No dataset_meta_json payload received from FL Server; "
            "falling back to the node's *_meta.json in MinIO."
        )
        global_vars["dataset_meta"] = None
        # Fallback: the task's assigned paths (set in filter_clients_if) include
        # the *_meta.json object; hydrate DatasetMeta straight from MinIO so the
        # client is not left without a schema if the server did not push one.
        assigned = global_vars.get("assigned_dataset_paths", [])
        meta_paths = [p for p in assigned if p.lower().endswith(".json")]
        if meta_paths:
            stages._hydrate_dataset_meta(meta_paths[0], global_vars)

        # Proceed to orchestrate the standard MLflow tracking parameters
    stages.set_run_config(
        experiment_id=content["experiment_id"],
        parent_run_id=content["parent_run_id"],
        node_name=global_vars["node_name"],
        model_name=content["model_name"],
        model_version=content["model_version"],
        data_path=global_vars["data_path"],
    )
    return responses_utils.create_success_response(msg, "set run config successfully")


def clean_config_if(msg: Message) -> Message:
    stages.clean_current_config()
    return responses_utils.create_success_response(msg, "cleaned config successfully")


def filter_clients_if(msg: Message, global_vars: dict) -> Message:
    """Participate iff the catalogue assigned this node data for the task use case.

    The use case is defined at dataset-upload time (e.g. ``"AML-1"``, ``"SCD-3"``);
    the server broadcasts the ``{node_name: [minio_url, ...]}`` map and each client
    selects its own slice by ``NODE_NAME``.
    """
    use_case, node_dataset_map = recordset.read_filter_clients_recordset(msg.content)

    node_name = global_vars["node_name"]
    assigned_paths = node_dataset_map.get(node_name, [])
    participate = bool(assigned_paths)

    if participate:
        global_vars["use_case"] = use_case
        global_vars["assigned_dataset_paths"] = assigned_paths
        logger.info(
            "Participating in task",
            node_name=node_name,
            use_case=use_case,
            datasets=assigned_paths,
        )
    else:
        logger.info(
            "Declining task: no datasets assigned for use case",
            node_name=node_name,
            use_case=use_case,
        )

    return responses_utils.create_participate_response(msg, participate)


def get_peer_discovery_if(msg: Message, global_vars: dict) -> Message:
    config = recordset.read_discover_peers_recordset(msg.content)
    logger.info(f"DISCOVER CONFIG → {config}")

    discovery_service = peer_discovery.PeerDiscovery(
        client_id=global_vars["node_name"],
        grpc_port=config['client_port'],
        min_peers=config['min_peers'],
        max_discovery_time=config['max_discovery_time'], )

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


def evaluate_model_if(msg: Message, global_vars: dict) -> Message:
    latest_parameters = recordset.read_evaluate_recordset(msg.content)
    stages.set_model_parameters(latest_parameters, global_vars)
    metrics = stages.evaluate_model(global_vars)
    return responses_utils.create_evaluate_response(msg, metrics)


def generate_synthetic_data_if(msg: Message, global_vars: dict) -> Message:
    """Handle a QUERY message requesting synthetic data generation.

    Reads ``n_samples`` from the incoming recordset, delegates to
    ``stages.generate_synthetic_data``, and returns the CSV payload
    in a ``ConfigsRecord`` under the key ``synthetic_csv``.
    """
    from flwr.common import ConfigsRecord, RecordSet, DEFAULT_TTL

    n_samples = recordset.read_generate_synthetic_data_recordset(msg.content)
    csv_str = stages.generate_synthetic_data(n_samples, global_vars)

    configs = ConfigsRecord({"success": True, "synthetic_csv": csv_str})
    rs = RecordSet(configs_records={"config": configs})
    return msg.create_reply(content=rs, ttl=DEFAULT_TTL)


def validate_synthetic_data_if(msg: Message, global_vars: dict) -> Message:
    """Handle a QUERY message requesting synthetic data validation.

    Reads ``synthetic_csv`` from the incoming recordset, delegates to
    ``stages.validate_synthetic_data``, and returns ``utility_report``
    and ``privacy_report`` HTML strings in the reply ``ConfigsRecord``.
    """
    from flwr.common import ConfigsRecord, RecordSet, DEFAULT_TTL

    csv_str = recordset.read_validate_synthetic_data_recordset(msg.content)
    reports = stages.validate_synthetic_data(csv_str, global_vars)

    configs = ConfigsRecord({
        "success": True,
        "utility": reports["utility"],
        "privacy": reports["privacy"],
    })
    rs = RecordSet(configs_records={"config": configs})
    return msg.create_reply(content=rs, ttl=DEFAULT_TTL)


#def generate_synthetic_if(msg: Message, global_vars: dict) -> Message:
#    try:
#        params = recordset.read_generate_synthetic_recordset(msg.content)
#    except Exception as e:
#        return create_error_response(msg, f"Invalid generate_synthetic recordset: {e}")

#    try:
#        result = stages.generate_synthetic_and_upload(
#            global_vars,
#            params["n_samples"],
#            params["sdg_model_name"],
#            params.get("dataset_meta_json", ""),
#            params.get("upload_to_s3", True),
#            params.get("s3_bucket", "synthetic-data"),
#            params.get("create_s3_bucket_if_missing", False),
#            params.get("real_data_partition", "test"),
#        )
#        return create_generate_synthetic_response(msg, result)
#    except Exception as e:
#        return create_error_response(msg, f"Synthetic generation failed: {e}")
