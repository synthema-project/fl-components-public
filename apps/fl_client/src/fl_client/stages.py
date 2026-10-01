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

from model_store_interface import DatasetPartition
from fl_client.metadata_contract import parse_metadata, variables, documents, evaluation_settings
from fl_client.data_contract import load_partitions, prepare_partitions
from fl_client.reporting_policy import utility_reporting_enabled, full_reporting_enabled

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
    mlflow_client.clean_current_config("FAILED")
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

    evaluation = json.loads(evaluation_config_json) if evaluation_config_json else None
    partitions, schema = load_partitions(dataset_paths, download_object, evaluation)
    if full_reporting_enabled():
        if not metadata_json or not evaluation_config_json:
            raise ValueError('Full validation requires the frozen central metadata/evaluation pair')
        schema = parse_metadata(json.loads(metadata_json), evaluation)
    # Catalogue governance metadata is not a feature schema. Reset per-task state.
    global_vars["model_meta"] = schema
    global_vars["prepared_partitions"] = {}
    if schema is None and metadata_json:
        payload = json.loads(metadata_json)
        if isinstance(payload, dict) and "variables" in payload:
            global_vars["model_meta"] = parse_metadata(payload, evaluation)
    global_vars["data_partitions"] = partitions
    global_vars["data"] = partitions["train"]
    global_vars["dataset"] = partitions["train"]
    global_vars["use_case"] = use_case


def prepare_data(global_vars: dict) -> None:
    learner = global_vars["local_learner"]
    metadata = global_vars.get("model_meta")
    cfg = evaluation_settings(metadata).get("aml_utility")
    if full_reporting_enabled() and not {"train", "val", "test"}.issubset(global_vars["data_partitions"]):
        raise ValueError("Full reporting requires independent train, val and test partitions from ingestion")
    if utility_reporting_enabled(metadata):
        from fl_client.data_contract import ensure_utility_holdout
        global_vars["data_partitions"] = ensure_utility_holdout(global_vars["data_partitions"], cfg)
    global_vars["data"] = global_vars["data_partitions"]["train"]
    global_vars["dataset"] = global_vars["data_partitions"]["train"]
    if _accepts_parameter(learner.prepare_data, "dataset_meta"):
        prepared = prepare_partitions(
            learner, global_vars["data_partitions"], global_vars.get("model_meta")
        )
        global_vars["prepared_partitions"] = prepared
        global_vars["data"] = prepared["train"]
    else:
        result = learner.prepare_data(data=global_vars["data"])
        if result is not None:
            global_vars["data"] = result


def load_model(global_vars: dict) -> None:
    model = mlflow_client.load_model()
    global_vars["model"] = model
    python_model = model.unwrap_python_model()
    payload = mlflow_client.load_dataset_meta(model)
    if payload is not None:
        registered_meta = parse_metadata(payload) if global_vars.get("model_meta") is None else global_vars["model_meta"]
        # Reporting-enabled models define the feature/endpoint contract and
        # holdout policy. Older catalogue metadata must not disable evaluation.
        if global_vars.get("model_meta") is None or evaluation_settings(registered_meta).get("aml_utility"):
            global_vars["model_meta"] = registered_meta
    factory = python_model.create_local_learner
    metadata = global_vars.get("model_meta")
    learner = factory(dataset_meta=metadata) if metadata is not None and _accepts_parameter(factory, "dataset_meta") else factory()
    if _accepts_parameter(learner.prepare_data, "dataset_meta") and metadata is None:
        raise ValueError("Model requires DatasetMeta, but neither dataset nor registered model provides a feature schema")
    global_vars["local_learner"] = learner


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
        validation = global_vars.get("prepared_partitions", {}).get("val")
        if validation is None:
            raise ValueError("Model evaluation requires the validation partition")
        metrics = evaluate(validation, dataset_meta)
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
        use_case=global_vars.get("use_case"),
    )
    return model_info


def generate_synthetic_data(n_samples: int, global_vars: dict, pool_id: str) -> dict[str, str]:
    from fl_client.utils.minio_loader import publish_pending_pool

    learner = global_vars["local_learner"]
    dataset_meta = global_vars.get("model_meta")
    generator = getattr(learner, "generate_synthetic_data", None)
    if generator is None or dataset_meta is None:
        raise RuntimeError("The loaded model does not support metadata-aware synthetic data generation")
    synthetic_data = generator(n_samples, dataset_meta)
    if not isinstance(synthetic_data, pd.DataFrame) or len(synthetic_data) != n_samples:
        raise ValueError("Generator did not return the requested number of synthetic rows")
    expected = [variable.name for variable in variables(dataset_meta)]
    if synthetic_data.columns.has_duplicates or not len(synthetic_data.columns):
        raise ValueError("Synthetic data requires unique, nonempty columns")
    if set(synthetic_data.columns) - set(expected):
        raise ValueError("Synthetic data contains columns absent from the model metadata")
    omitted = sorted(set(expected) - set(synthetic_data.columns))
    if omitted and full_reporting_enabled():
        raise ValueError("Full validation requires all configured variables in the generated pool")
    if omitted:
        logger.warning("Model did not generate all metadata variables", omitted_variables=omitted)
    expected = [name for name in expected if name in synthetic_data.columns]
    metadata, evaluation = documents(dataset_meta)
    source_meta_hash = metadata.pop("meta_hash", None)
    metadata["variables"] = [v for v in metadata["variables"] if v["name"] in expected]
    metadata["use_case"] = global_vars["use_case"]
    # Preserve MSI types and add the storage-interface representation.
    storage_types = {"numeric": "Number", "integer": "Integer",
                     "categorical": "Categorical", "binary": "Binary",
                     "datetime": "Datetime", "text": "Text"}
    for variable in metadata["variables"]:
        if "datatype" not in variable:
            variable["datatype"] = storage_types[variable["type"]]
    evaluation = dict(evaluation)
    evaluation["generation"] = {"omitted_variables": omitted, "validation_performed": False,
                                "source_meta_hash": source_meta_hash}
    result = publish_pending_pool(
        synthetic_data[expected], metadata, evaluation,
        use_case=global_vars["use_case"], pool_id=pool_id,
    )
    result["utility_reporting_enabled"] = utility_reporting_enabled(dataset_meta) or full_reporting_enabled()
    return result


def validate_synthetic_pool(pool_id: str, object_uri: str, use_case: str,
                            global_vars: dict) -> dict[str, Any]:
    if full_reporting_enabled():
        from fl_client.validation_runner import generate_reports
        return generate_reports(pool_id, object_uri, use_case, global_vars)
    if not utility_reporting_enabled(global_vars.get("model_meta")):
        raise ValueError("Utility reporting is disabled for this deployment or model")
    from fl_client.utils.minio_loader import upload_pool_utility_report

    metadata = global_vars.get("model_meta")
    if use_case.upper() != "AML" or metadata is None:
        raise ValueError("AML utility reporting requires AML model metadata")
    holdout = global_vars.get("data_partitions", {}).get("test")
    if holdout is None:
        raise ValueError("AML utility reporting requires an independent test/holdout partition")
    synthetic = pd.read_csv(download_object(object_uri))
    reporting_python = os.getenv("AML_UTILITY_PYTHON")
    if not reporting_python and os.getenv("VALIDATION_PROFILE") == "aml":
        reporting_python = "/opt/aml-utility/bin/python"
    if os.getenv("VALIDATION_PROFILE") == "none" and not reporting_python:
        raise ValueError("This client image was built without AML reporting; deploy an aml image")
    if reporting_python:
        import tempfile
        import subprocess
        from pathlib import Path
        with tempfile.TemporaryDirectory(prefix="aml-report-") as work:
            work = Path(work)
            synthetic.to_csv(work / "synthetic.csv", index=False)
            holdout.to_csv(work / "holdout.csv", index=False)
            (work / "metadata.json").write_text(metadata.model_dump_json())
            try:
                subprocess.run([
                    reporting_python, "-m", "model_store_interface.aml_utility",
                    "--synthetic", str(work / "synthetic.csv"),
                    "--holdout", str(work / "holdout.csv"),
                    "--metadata", str(work / "metadata.json"),
                    "--output", str(work / "report.html"),
                    "--node", global_vars["node_name"],
                ], check=True, timeout=1500, capture_output=True, text=True)
            except subprocess.CalledProcessError as error:
                # Do not send raw data or validator stdout over Flower.
                logger.error("AML report process failed", returncode=error.returncode)
                raise RuntimeError("AML utility report failed; verify configured columns, event values and survival durations") from error
            report = (work / "report.html").read_text()
    else:
        from model_store_interface.aml_utility import generate_aml_utility_report
        report = generate_aml_utility_report(synthetic, holdout, metadata, global_vars["node_name"])
    report_uri = upload_pool_utility_report(report, object_uri=object_uri,
                                           pool_id=pool_id, node_name=global_vars["node_name"])
    return {"node_name": global_vars["node_name"], "report_uri": report_uri,
            "metrics": {"provider": "sd_quality", "scope": "aml_utility",
                        "holdout_rows": len(holdout), "synthetic_rows": len(synthetic),
                        "privacy_evaluated": False}}


def clean_current_config(status: str = "FINISHED") -> None:
    mlflow_client.clean_current_config(status)

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
