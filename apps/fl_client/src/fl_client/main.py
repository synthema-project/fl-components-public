from flwr.client import ClientApp
from flwr.common import Context, Message
import structlog
from fl_client.config import MLFLOW_URL, NODE_NAME, DATA_PATH

logger = structlog.get_logger()
from fl_client.interface import (
    clean_config_if,
    filter_clients_if,
    generate_synthetic_data_if,
    get_parameters_if,
    load_data_if,
    load_model_if,
    prepare_data_if,
    set_parameters_if,
    set_run_config_if,
    train_model_fixed_clipping_if,
    train_model_if,
    upload_model_if,
    get_peer_discovery_if,
    connect_peers_if,
    start_P2P_server_if,
    train_model_SMPC_if,
    set_smpc_enabled_if,
    set_dp_config_if,
    evaluate_model_if,
    validate_synthetic_data_if,
)

from interfaces.mlflow_client import setup_mlflow

global_vars = {
    "data": None,
    "local_learner": None,
    "model": None,
    "model_meta": None,
    "use_case": None,
    "assigned_dataset_paths": None,
    "node_name": NODE_NAME,
    "data_path": DATA_PATH,
    "SMPC_enabled": False,
    "peers": None,
    "peer_stubs": None,
    "smpc_client": None,
    "discovered_peers": None,
    "discovered_peers_ids": None,
    "dp_config": None,
    "dataset_meta_json": None,
    "dataset_meta": None,
}

setup_mlflow(MLFLOW_URL)
app = ClientApp()

@app.query()
def query(msg: Message, ctx: Context) -> Message:
    global global_vars

    mode = msg.content.configs_records["config"]["mode"]
    if not isinstance(mode, str):
        raise TypeError()

    if mode == "filter_clients":
        return filter_clients_if(msg, global_vars)

    elif mode == "load_data":
        return load_data_if(msg, global_vars)

    elif mode == "load_model":
        return load_model_if(msg, global_vars)

    elif mode == "get_parameters":
        return get_parameters_if(msg, global_vars)

    elif mode == "set_parameters":
        return set_parameters_if(msg, global_vars)

    elif mode == "set_run_config":
        return set_run_config_if(msg, global_vars)

    elif mode == "set_smpc_enabled":
        return set_smpc_enabled_if(msg, global_vars)
    elif mode == "set_dp_config":
        return set_dp_config_if(msg, global_vars)

    elif mode == "prepare_data":
        return prepare_data_if(msg, global_vars)

    elif mode == "upload_model":
        return upload_model_if(msg, global_vars)

    elif mode == "clean_config":
        return clean_config_if(msg)
    # start grpc server for all clients
    elif mode == "start_P2P_server":
        return start_P2P_server_if(msg, global_vars)
    elif mode == "discover_peers":
        return get_peer_discovery_if(msg, global_vars)
    # P2P network connection
    elif mode == "connect_peers":
        # returns the peers and peer_stubs
        return connect_peers_if(msg, global_vars)
    #elif mode == "generate_synthetic":
    #    return generate_synthetic_if(msg, global_vars)
    elif mode == "generate_synthetic_data":
        return generate_synthetic_data_if(msg, global_vars)

    elif mode == "validate_synthetic_data":
        return validate_synthetic_data_if(msg, global_vars)

    else:
        raise NotImplementedError(f"Unknown mode: {mode}")


@app.train()
def train(msg: Message, ctx: Context) -> Message:
    global global_vars

    dp_config = global_vars.get("dp_config")
    smpc_enabled = global_vars.get("SMPC_enabled", False)

    # Check for incompatible configuration
    if dp_config and dp_config.get("dp_mode") == "central_dp" and smpc_enabled:
        raise RuntimeError("Cannot use both SMPC and DP in central_dp mode. Please disable one of them.")

    # Route to appropriate training function
    if smpc_enabled:
        logger.info("Training with SMPC enabled")
        return train_model_SMPC_if(msg, global_vars)
    elif dp_config and dp_config.get("dp_mode") == "central_dp":
        return train_model_fixed_clipping_if(msg, global_vars)
    else:
        return train_model_if(msg, global_vars)


@app.evaluate()
def evaluate(msg: Message, ctx: Context) -> Message:
    global global_vars
    return evaluate_model_if(msg, global_vars)

