import inspect
import io
from typing import Any, List, Dict

import grpc
import numpy as np
import pandas as pd
import structlog
from flwr.common import MetricsRecord, ParametersRecord
import structlog
from fl_client import smpc_pb2
from fl_client.SMPCClient import SMPCClient
from fl_client.utils.minio_loader import download_object, upload_file
import json
import tempfile
import os
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


#def load_data(
#        use_case: str,
#        dataset_paths: list[str],
#        global_vars: dict,
#):
#    if not dataset_paths:
#        raise ValueError(
#            f"No dataset paths received for node {global_vars['node_name']}"
#        )

#    dataframes = []

#    for object_path in dataset_paths:
        # Support partitioned dataset paths of the form "partition:path",
        # e.g. "train:s3://bucket/train.csv". If no partition prefix is
        # provided we keep old behavior and concatenate everything.
#        if ":" in object_path and not object_path.startswith("s3://"):
            # interpret as partition:path
#            partition, path = object_path.split(":", 1)
#            local_file = download_object(path)
#            df = pd.read_csv(local_file)
#            # store per-partition in dataset_partitions dict
#            if "dataset_partitions" not in global_vars:
#                global_vars["dataset_partitions"] = {}
#            global_vars["dataset_partitions"][partition] = df
#        else:
#            local_file = download_object(object_path)
#            df = pd.read_csv(local_file)
#            dataframes.append(df)

    # If we collected unlabeled dataframes, concatenate them as the default
    # dataset. Also keep per-partition mapping available in global_vars.
#    if dataframes:
#        dataset = pd.concat(dataframes, ignore_index=True)
#        global_vars["dataset"] = dataset
#    else:
#        # If no unlabeled dataframes were provided, try to build a global
#        # dataset from available partitions (concatenate all partitions).
#        partitions = global_vars.get("dataset_partitions", {})
#        if partitions:
#            global_vars["dataset"] = pd.concat(list(partitions.values()), ignore_index=True)
#        else:
#            global_vars["dataset"] = pd.DataFrame()
#    global_vars["use_case"] = use_case


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

def generate_synthetic_and_upload(global_vars: dict,
                                  n_samples: int,
                                  sdg_model_name: str,
                                  dataset_meta_json: str = "",
                                  upload_to_s3: bool = True,
                                  s3_bucket: str = "synthetic-data-for-inference",
                                  create_s3_bucket_if_missing: bool = False,
                                  real_data_partition: str = "test") -> dict:
    """Generate synthetic data using local_learner, run validations and upload artifacts to MinIO.

    Returns a dictionary with uploaded object URIs and local paths.
    """
    local_learner = global_vars.get("local_learner")
    node_name = os.environ.get("NODE_NAME") or global_vars.get("node_name")
    use_case = global_vars.get("use_case") or "unknown"
    node_id = node_name

    if local_learner is None:
        raise RuntimeError("Local learner not initialized on client")

    # Prepare dataset_meta
    dataset_meta_obj = None
    try:
        if dataset_meta_json:
            # Try to import DatasetMeta from model_store_interface
            try:
                from model_store_interface.core.schemas.dataset_meta import DatasetMeta
                dataset_meta_obj = DatasetMeta.model_validate_json(dataset_meta_json)
            except Exception:
                # Fallback to plain dict
                dataset_meta_obj = json.loads(dataset_meta_json)
        else:
            # Build a simple metadata inference from local dataset
            dataset = global_vars.get("dataset")
            if dataset is None:
                raise RuntimeError("No local dataset available to infer metadata")
            # Minimal metadata: title and variables
            cols = list(dataset.columns)
            variables = [{"name": c, "type": "numeric"} for c in cols]
            dataset_meta_obj = {"title": f"inferred_{use_case}", "variables": variables, "use_case": use_case}
    except Exception as e:
        raise RuntimeError(f"Failed to prepare dataset_meta: {e}")

    # Generate synthetic data
    try:
        synthetic_df = local_learner.generate_synthetic_data(n_samples, dataset_meta_obj)
    except Exception as e:
        raise RuntimeError(f"Local learner failed to generate synthetic data: {e}")

    # Validation
    utility_report_html = None
    privacy_report_html = None
    try:
        from model_store_interface.validation import validate_utility_report, validate_privacy_report
        # Prefer using a specific real-data partition for validation if available.
        partitions = global_vars.get("dataset_partitions", {})
        if real_data_partition and partitions and real_data_partition in partitions:
            real_data = partitions[real_data_partition]
        else:
            real_data = global_vars.get("dataset")
        # These functions expect DatasetMeta model in many cases; try to pass object or dict
        utility_report_html = validate_utility_report(synthetic_df, real_data, dataset_meta_obj, node_id)
        privacy_report_html = validate_privacy_report(synthetic_df, real_data, dataset_meta_obj, node_id)
    except Exception as e:
        # If either validation fails, embed the error in the HTML placeholder
        if utility_report_html is None:
            utility_report_html = f"<html><body><pre>Utility validation failed: {e}</pre></body></html>"
        if privacy_report_html is None:
            privacy_report_html = f"<html><body><pre>Privacy validation failed: {e}</pre></body></html>"

    # Persist artifacts locally
    tmpdir = tempfile.mkdtemp(prefix="synthetic_")
    data_fname = os.path.join(tmpdir, "data.csv")
    meta_fname = os.path.join(tmpdir, "dataset_meta.json")
    util_fname = os.path.join(tmpdir, "utility_report.html")
    priv_fname = os.path.join(tmpdir, "privacy_report.html")

    synthetic_df.to_csv(data_fname, index=False)

    # dataset_meta_obj may be pydantic; ensure serializable
    try:
        if hasattr(dataset_meta_obj, 'model_dump'):
            meta_json = dataset_meta_obj.model_dump_json(exclude_none=True)
        else:
            meta_json = json.dumps(dataset_meta_obj)
    except Exception:
        meta_json = json.dumps(dataset_meta_obj)

    with open(meta_fname, 'w') as f:
        f.write(meta_json)

    with open(util_fname, 'w') as f:
        f.write(utility_report_html or "")

    with open(priv_fname, 'w') as f:
        f.write(privacy_report_html or "")

    result = {
        "local_paths": {
            "data": data_fname,
            "meta": meta_fname,
            "utility_report": util_fname,
            "privacy_report": priv_fname,
        },
        "node_name": node_name,
        "s3_uris": {}
    }

    # Upload to S3/MinIO if requested
    if upload_to_s3:
        # Build prefix
        prefix = f"{use_case}/{sdg_model_name}/{node_name}"
        data_key = f"{prefix}/data.csv"
        meta_key = f"{prefix}/dataset_meta.json"
        util_key = f"{prefix}/utility_report.html"
        priv_key = f"{prefix}/privacy_report.html"

        try:
            upload_file(data_fname, s3_bucket, data_key)
            upload_file(meta_fname, s3_bucket, meta_key)
            upload_file(util_fname, s3_bucket, util_key)
            upload_file(priv_fname, s3_bucket, priv_key)

            result["s3_uris"] = {
                "data": f"s3://{s3_bucket}/{data_key}",
                "meta": f"s3://{s3_bucket}/{meta_key}",
                "utility_report": f"s3://{s3_bucket}/{util_key}",
                "privacy_report": f"s3://{s3_bucket}/{priv_key}",
            }
        except Exception as e:
            # Upload failed, include error in result
            result["s3_upload_error"] = str(e)

    return result
