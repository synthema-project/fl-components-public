from typing import Any, List, Dict
import inspect
import json
import os
import numpy as np
import grpc
import pandas as pd
from flwr.common import MetricsRecord, ParametersRecord
import structlog
from fl_client.config import DATA_PATH
from fl_client.config import DATA_PATH_AML

from interfaces import mlflow_client
from fl_client.SMPCClient import SMPCClient
from fl_client import smpc_pb2
from fl_client.utils.minio_loader import download_object
from fl_client.utils.minio_loader import upload_dataframe, upload_json, upload_html, download_object

try:
    from model_store_interface.core.dataset_partition import DatasetPartition
    from model_store_interface.metadata.models.dataset_metadata import DatasetMetadata
    from model_store_interface.metadata.models.model import SynthemaDatasetModel
except ImportError:  # pragma: no cover - compatibility with legacy images
    DatasetPartition = None
    DatasetMetadata = None
    SynthemaDatasetModel = None

logger = structlog.get_logger()


def _accepts_parameter(callable_obj: Any, parameter_name: str) -> bool:
    """Return whether a callable can receive the named keyword argument."""
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == parameter_name
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def set_run_config(
    experiment_id: str,
    parent_run_id: str,
    node_name: str,
    model_name: str,
    model_version: int,
    data_path: str,
) -> None:
    if mlflow_client.mlflow.active_run():
        mlflow_client.mlflow.end_run("FAILED")
    child_run_id = mlflow_client.create_child_run(
        experiment_id, parent_run_id, node_name
    )
    mlflow_client.set_current_config(
        experiment_id, parent_run_id, child_run_id, model_name, model_version
    )
    mlflow_client.set_dataset_signature(node_name, data_path, child_run_id)


def load_data(
    use_case: str,
    dataset_paths: list[str] | dict,
    global_vars: dict | None = None,
    metadata_json: str | None = None,
    evaluation_config_json: str | None = None,
):
    # Keep the old local-simulation call shape (load_data(use_case, vars))
    # usable while the Flower path supplies catalogue object paths.
    if global_vars is None and isinstance(dataset_paths, dict):
        global_vars = dataset_paths
        if use_case == "iris":
            dataset_paths = [DATA_PATH]
        elif use_case == "AML":
            dataset_paths = [DATA_PATH_AML]
        else:
            raise NotImplementedError(f"No local dataset configured for use case '{use_case}'")

    if global_vars is None:
        raise TypeError("global_vars is required")
    if not dataset_paths:
        raise ValueError(
            f"No dataset paths received for node {global_vars['node_name']}"
        )

    partition_paths = dataset_paths if isinstance(dataset_paths, dict) else {}
    paths_to_load = (partition_paths.get("train") or partition_paths.get("val") or partition_paths.get("test")) \
        if partition_paths else dataset_paths
    if isinstance(paths_to_load, str):
        paths_to_load = [paths_to_load]
    dataframes = []

    for object_path in paths_to_load:
        local_file = object_path if object_path.startswith(("/", ".")) else download_object(object_path)
        df = pd.read_csv(local_file)
        dataframes.append(df)

    dataset = pd.concat(dataframes, ignore_index=True)

    global_vars["data"] = dataset
    global_vars["dataset"] = dataset
    global_vars["data_partitions"] = {
        role: pd.read_csv(path if path.startswith(("/", ".")) else download_object(path))
        for role, path in partition_paths.items() if role != "schema"
    } if partition_paths else {}
    global_vars["use_case"] = use_case
    if metadata_json:
        if SynthemaDatasetModel is not None and evaluation_config_json:
            global_vars["model_meta"] = SynthemaDatasetModel.from_json(
                metadata_json=json.loads(metadata_json),
                evaluation_configuration_json=json.loads(evaluation_config_json),
            )
        else:
            metadata_payload = json.loads(metadata_json)
            global_vars["model_meta"] = (
                DatasetMetadata.from_health_dcat_json(metadata_payload)
                if DatasetMetadata is not None else metadata_payload
            )



def prepare_data(global_vars: dict) -> None:
    local_learner = global_vars["local_learner"]
    data = global_vars["data"]
    dataset_meta = global_vars.get("model_meta")
    if (
        dataset_meta is not None
        and DatasetPartition is not None
        and _accepts_parameter(local_learner.prepare_data, "dataset_meta")
    ):
        prepared = local_learner.prepare_data(
            data=data,
            partition=DatasetPartition.TRAIN,
            dataset_meta=dataset_meta,
        )
        if prepared is not None:
            global_vars["data"] = prepared
    else:
        local_learner.prepare_data(data=data)


def load_model(
    global_vars: dict,
) -> None:
    mlf_model = mlflow_client.load_model()
    global_vars["model"] = mlf_model
    python_model = mlf_model.unwrap_python_model()
    if global_vars.get("model_meta") is None and hasattr(python_model, "get_metadata_documents"):
        try:
            metadata_json, evaluation_config_json = python_model.get_metadata_documents()
        except RuntimeError:
            pass
        else:
            if SynthemaDatasetModel is not None:
                global_vars["model_meta"] = SynthemaDatasetModel.from_json(
                    metadata_json=metadata_json,
                    evaluation_configuration_json=evaluation_config_json,
                )
    factory = python_model.create_local_learner
    dataset_meta = global_vars.get("model_meta")
    if dataset_meta is not None and _accepts_parameter(factory, "dataset_meta"):
        global_vars["local_learner"] = factory(dataset_meta=dataset_meta)
    else:
        global_vars["local_learner"] = factory()


def train_model(
    global_vars: dict,
    current_global_iter: int,
) -> MetricsRecord:
    local_learner = global_vars["local_learner"]
    dataset_meta = global_vars.get("model_meta")
    train_round = getattr(local_learner, "train_round", None)
    if dataset_meta is not None and train_round is not None and _accepts_parameter(train_round, "dataset_meta"):
        metrics = train_round(global_vars["data"], dataset_meta)
    else:
        metrics = local_learner.train()
    if not isinstance(metrics, MetricsRecord):
        raise TypeError(f"Unexpected metrics type: {type(metrics)}")
    mlflow_client.log_metrics(metrics, current_global_iter)
    return metrics


def evaluate_model(global_vars: dict) -> MetricsRecord:
    local_learner = global_vars["local_learner"]
    dataset_meta = global_vars.get("model_meta")
    evaluate = getattr(local_learner, "evaluate")
    if dataset_meta is not None and _accepts_parameter(evaluate, "dataset_meta"):
        metrics = evaluate(global_vars["data"], dataset_meta)
    else:
        metrics = evaluate()
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

def set_dp_config(dp_params:dict, global_vars:dict):
    global_vars["dp_config"] = dp_params



def upload_model(
    latest_parameters: ParametersRecord,
    global_vars: dict,
) -> dict[str, Any]:
    local_learner = global_vars["local_learner"]
    set_model_parameters(latest_parameters, global_vars)
    model_info: dict[str, Any] = mlflow_client.upload_final_state(
        local_learner,
        dataset_meta=global_vars.get("model_meta"),
    )
    return model_info


def generate_synthetic_data(n_samples: int, global_vars: dict) -> dict[str, str]:
    learner = global_vars["local_learner"]
    dataset_meta = global_vars.get("model_meta")
    generator = getattr(learner, "generate_synthetic_data", None)
    if generator is None or dataset_meta is None:
        raise RuntimeError("The loaded model does not support metadata-aware synthetic data generation")
    synthetic_data = generator(n_samples, dataset_meta)
    object_uri = upload_dataframe(
        synthetic_data,
        use_case=global_vars["use_case"],
        model_name=global_vars.get("model_name", "sdg_model"),
        node_name=global_vars["node_name"],
    )
    metadata_json, evaluation_json = dataset_meta.to_json()
    return {
        "object_uri": object_uri,
        "metadata_uri": upload_json(
            metadata_json,
            use_case=global_vars["use_case"],
            model_name=global_vars.get("model_name", "sdg_model"),
            node_name=global_vars["node_name"],
            filename="dataset_meta.json",
        ),
        "evaluation_uri": upload_json(
            evaluation_json,
            use_case=global_vars["use_case"],
            model_name=global_vars.get("model_name", "sdg_model"),
            node_name=global_vars["node_name"],
            filename="evaluation_config.json",
        ),
    }


def validate_synthetic_pool(pool_id: str, object_uri: str, use_case: str,
                            global_vars: dict) -> dict[str, Any]:
    """Run the configured validator, or a clearly-labelled smoke validator.

    Private AML/SCD packages can later be exposed as ``module:callable`` via
    VALIDATION_PROVIDER. They receive (real_data, synthetic_data, use_case,
    dataset_meta) and must return ``(metrics, html)``.
    """
    synthetic = pd.read_csv(download_object(object_uri))
    real = global_vars.get("data_partitions", {}).get("train", global_vars.get("data"))
    provider = os.getenv("VALIDATION_PROVIDER", "placeholder")
    if provider != "placeholder":
        module_name, function_name = provider.split(":", 1)
        provider_fn = getattr(__import__(module_name, fromlist=[function_name]), function_name)
        metrics, html = provider_fn(real, synthetic, use_case, global_vars.get("model_meta"))
    else:
        common = sorted(set(real.columns) & set(synthetic.columns))
        metrics = {"provider": "placeholder", "column_overlap": len(common) / max(len(real.columns), 1),
                   "synthetic_rows": int(len(synthetic)), "real_rows": int(len(real))}
        html = (f"<html><body><h1>Placeholder validation report</h1>"
                f"<p>Use case: {use_case}</p><pre>{json.dumps(metrics, indent=2)}</pre>"
                "<p>Replace VALIDATION_PROVIDER with the private validator.</p></body></html>")
    report_uri = upload_html(html, use_case=use_case,
                             model_name=global_vars.get("model_name", "sdg_model"),
                             node_name=global_vars["node_name"],
                             filename=f"{pool_id}_{global_vars['node_name']}.html")
    return {"node_name": global_vars["node_name"], "metrics": metrics, "report_uri": report_uri}

def clean_current_config() -> None:
    mlflow_client.clean_current_config()

def start_P2P_server(client_id: int, port: int, global_vars: dict) -> SMPCClient:
    smpc_client = SMPCClient(client_id, port)
    global_vars["smpc_client"] = smpc_client
    return smpc_client

def set_discovered_peers(discovered_peers: List[str], discovered_peers_ids: List[str], global_vars: dict) -> List[str]:
    global_vars['discovered_peers'] = discovered_peers
    global_vars['discovered_peers_ids'] = discovered_peers_ids
    return discovered_peers

def set_SMPC_enabled(SMPC_enabled: bool, global_vars: dict) -> bool:
    global_vars["SMPC_enabled"] = SMPC_enabled
    return SMPC_enabled

def connect_peers(discovered_peers, global_vars: dict) -> dict[str, str]:
    smpc_client = global_vars["smpc_client"]
    peer_stubs = smpc_client.connect_to_peers(discovered_peers)
    global_vars["peer_stubs"] = peer_stubs
    return peer_stubs

def send_shares_to_peers(secret_shares: Dict[str, List[np.ndarray]],
                         peer_stubs: Dict[str, Any],
                         client_id: str):
    """
    Sends secret shares to all peer clients.

    Parameters:
    secret_shares (Dict[str, List[np.ndarray]]): Dictionary mapping peer node names to lists of numpy array shares.
    peer_stubs (Dict[str, gRPC Stub]): Dictionary mapping peer node names to gRPC stubs for communication.
    client_id (str): The ID of the client sending the shares.
    """
    for peer_node_name, stub in peer_stubs.items():
        node_names = list(secret_shares.keys())
        shares = secret_shares.get(node_names[peer_node_name])
        if shares is None:
            logger.error(f"No shares found for peer {node_names[peer_node_name]}")
            continue

        share_protos = []
        # for idx, matrix in enumerate(shares):
        #     flattened_data = matrix.flatten()
        #     share_proto = smpc_pb2.Share(
        #         name=f"param_{idx}",  # Add a simple name for tracking
        #         data=flattened_data.astype(np.float32).tobytes(),
        #         shape=list(matrix.shape)
        #     )
        #     share_protos.append(share_proto)
        for name, matrix in shares.items():
            flattened_data = matrix.flatten()
            share_proto = smpc_pb2.Share(
                name=name,  # Use the actual name like 'linear.weight'
                data=flattened_data.astype(np.float32).tobytes(),
                shape=list(matrix.shape)
            )
            share_protos.append(share_proto)

        request = smpc_pb2.SharesRequest(
            client_id=client_id,
            shares=share_protos
        )

        try:
            response = stub.SendShares(request)
            if response.status == "ACK":
                logger.info(f"Client {client_id} successfully sent shares to peer {node_names[peer_node_name]}")
            else:
                logger.error(f"Unexpected response from peer {node_names[peer_node_name]}: {response.status}")
        except grpc.RpcError as e:
            logger.error(f"Failed to send shares to peer {node_names[peer_node_name]}: {e}")

def reconstruct_shares(local_shares: Dict, smpc_client: SMPCClient):
    try:
        # Wait for all shares from other clients
        expected_shares = len(smpc_client.peer_addresses)
        if expected_shares > 0:
            if not smpc_client.all_shares_received.wait(timeout=60):
                raise TimeoutError(f"Not all shares received for client {smpc_client.client_id}")
            print("-- ALL SHARES RECEIVED --")
            # Combine local and received shares properly with client IDs
            all_shares = {f"client_{i}": shares for i, shares in enumerate(
                [local_shares] + [smpc_client.received_shares[x] for x in smpc_client.received_shares.keys()]
            )}

            # Now reconstruct properly
            reconstructed_shares = smpc_client.reconstruct_model_weights(all_shares)

            # Reset for next round
            smpc_client.all_shares_received.clear()
            smpc_client.received_shares.clear()

            return reconstructed_shares
        else:
            # Solo client case
            return smpc_client.model.get_weights()
    except TimeoutError as e:
        logger.error(f"Client {smpc_client.client_id} failed to receive all shares: {e}")
        return None
