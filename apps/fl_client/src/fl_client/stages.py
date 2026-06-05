import inspect
import io
from typing import Any, List, Dict

import grpc
import numpy as np
import pandas as pd
import structlog
from flwr.common import MetricsRecord, ParametersRecord
from interfaces import mlflow_client
from model_store_interface import validation
from model_store_interface.core.dataset_partition import DatasetPartition
from model_store_interface.core.schemas.dataset_meta import DatasetMeta

from fl_client import smpc_pb2
from fl_client.SMPCClient import SMPCClient
from fl_client.utils.minio_loader import download_object

# Import the strict MLOps partitioning enum

logger = structlog.get_logger()


def validate_utility_report(*args, **kwargs):
    return validation.validate_utility_report(*args, **kwargs)


def validate_privacy_report(*args, **kwargs):
    return validation.validate_privacy_report(*args, **kwargs)


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
        use_case: str, dataset_paths: list[str], global_vars: dict
) -> None:
    """Loads and mathematically partitions the clinical tabular data."""
    if not dataset_paths:
        raise ValueError(
            f"No dataset paths received for node {global_vars['node_name']}"
        )

    logger.info(f"Loading real clinical data from {dataset_paths}...")

    dataframes = []
    for object_path in dataset_paths:
        local_file = download_object(object_path)
        df = pd.read_csv(local_file)
        dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)

    # Strict DevSecOps Partitioning (60% Train, 20% Val, 20% Test)
    # The fixed random_state guarantees deterministic splits across all Ubuntu edge nodes
    shuffled_data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_samples = len(shuffled_data)
    train_end = int(n_samples * 0.6)
    val_end = int(n_samples * 0.8)

    global_vars[DatasetPartition.TRAIN] = shuffled_data.iloc[:train_end].reset_index(drop=True)
    global_vars[DatasetPartition.VAL] = shuffled_data.iloc[train_end:val_end].reset_index(drop=True)
    global_vars[DatasetPartition.TEST] = shuffled_data.iloc[val_end:].reset_index(drop=True)

    # Legacy compatibility for older learners
    global_vars["data"] = data


def prepare_data(global_vars: dict) -> None:
    """Prepares dataset partitions using the local learner."""
    local_learner = global_vars["local_learner"]
    sig = inspect.signature(local_learner.prepare_data)

    if "partition" in sig.parameters:
        # Call prepare_data for each partition and save the prepared dataframes
        for partition in [DatasetPartition.TRAIN, DatasetPartition.VAL, DatasetPartition.TEST]:
            data = global_vars.get(partition)
            if data is not None:
                kwargs = {}
                if "data" in sig.parameters:
                    kwargs["data"] = data
                kwargs["partition"] = partition
                if "dataset_meta" in sig.parameters:
                    kwargs["dataset_meta"] = global_vars.get("dataset_meta")
                prepared = local_learner.prepare_data(**kwargs)
                global_vars[f"prepared_{partition.value}"] = prepared
    else:
        # Legacy compatibility - prepare only the train/raw data
        train_data = global_vars.get(DatasetPartition.TRAIN, global_vars.get("data"))
        local_learner.prepare_data(train_data)


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
    if hasattr(local_learner, "train_round"):
        train_data = global_vars.get("prepared_train", global_vars.get(DatasetPartition.TRAIN, global_vars["data"]))
        metrics = local_learner.train_round(
            data=train_data,
            dataset_meta=global_vars.get("dataset_meta")
        )
    elif hasattr(local_learner, "train"):
        metrics = local_learner.train()
    else:
        raise AttributeError("local_learner has neither 'train_round' nor 'train' method")

    if not isinstance(metrics, MetricsRecord):
        raise TypeError(f"Unexpected metrics type: {type(metrics)}")
    mlflow_client.log_metrics(metrics, current_global_iter)
    return metrics


def evaluate_model(global_vars: dict) -> MetricsRecord:
    local_learner = global_vars["local_learner"]
    if hasattr(local_learner, "evaluate"):
        sig = inspect.signature(local_learner.evaluate)
        if len(sig.parameters) == 0:
            metrics = local_learner.evaluate()
        else:
            val_data = global_vars.get("prepared_val", global_vars.get(DatasetPartition.VAL, global_vars["data"]))
            metrics = local_learner.evaluate(
                data=val_data,
                dataset_meta=global_vars.get("dataset_meta")
            )
    else:
        raise AttributeError("local_learner has no 'evaluate' method")

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


def set_dp_config(dp_params: dict, global_vars: dict):
    global_vars["dp_config"] = dp_params


def upload_model(
        latest_parameters: ParametersRecord,
        global_vars: dict,
) -> dict[str, Any]:
    local_learner = global_vars["local_learner"]
    set_model_parameters(latest_parameters, global_vars)
    model_info: dict[str, Any] = mlflow_client.upload_final_state(local_learner)
    return model_info


def clean_current_config() -> None:
    mlflow_client.clean_current_config()


def generate_synthetic_data(n_samples: int, global_vars: dict) -> str:
    """Generate synthetic data using the local learner and return it as a CSV string.

    Args:
        n_samples: Number of synthetic rows to generate.
        global_vars: Shared node state containing ``local_learner`` and ``dataset_meta``.

    Returns:
        CSV-encoded string of the synthetic DataFrame.

    Raises:
        ValueError: If ``dataset_meta`` is not present in ``global_vars``.
    """
    dataset_meta = global_vars.get("dataset_meta")
    if dataset_meta is None:
        raise ValueError("dataset_meta is not available in global_vars; cannot generate synthetic data")

    local_learner = global_vars["local_learner"]
    df: pd.DataFrame = local_learner.generate_synthetic_data(n_samples, dataset_meta)
    return df.to_csv(index=False)


def validate_synthetic_data(synthetic_csv: str, global_vars: dict) -> dict[str, str]:
    """
    Run utility and privacy validation against the node's TEST partition.
    Utilises the strictly typed DatasetMeta object to securely route plugins.
    """
    logger.info("Initiating Phase 4: Strict DCAT-AP post-training federated validation.")

    # EXTRACT HYDRATED OBJECT: No string parsing required
    meta: DatasetMeta | None = global_vars.get("dataset_meta")

    if not meta:
        logger.error("CRITICAL: No hydrated DatasetMeta found in global_vars. Aborting validation.")
        return {"utility": "", "privacy": ""}

    try:
        # 1. Parse the incoming generative 'Big Pool' payload
        synthetic_data = pd.read_csv(io.StringIO(synthetic_csv))

        # 2. Extract the local 20% hold-out real data partition
        real_data = global_vars[DatasetPartition.TEST]

        logger.info(
            f"Executing clinical validation plugins for domain {meta.use_case.value} "
            f"against {len(real_data)} real samples."
        )

        # 3. Execute DevSecOps validation plugins (aml-u, scd-u, sba-priv)
        utility_report = validate_utility_report(
            synthetic_data=synthetic_data,
            real_data=real_data,
            dataset_meta=meta,
            node_id=global_vars["node_name"]
        )

        privacy_report = validate_privacy_report(
            synthetic_data=synthetic_data,
            real_data=real_data,
            dataset_meta=meta,
            node_id=global_vars["node_name"]
        )

        return {
            "utility": utility_report,
            "privacy": privacy_report
        }

    except Exception as e:
        logger.error(f"Catastrophic failure during synthetic data validation: {e}")
        return {"utility": "", "privacy": ""}


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
