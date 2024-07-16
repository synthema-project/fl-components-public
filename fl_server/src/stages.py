from typing import Any
import mlflow
from flwr.common import Message, ParametersRecord, MetricsRecord

from common.utils.src.mlflow_utils import load_mlflow_model


def load_model(
    model_name: str,
    model_version: int,
    mlflow_client: mlflow.MlflowClient,
    global_vars: dict[str, Any],
) -> None:
    mlf_model, mlf_model_meta = load_mlflow_model(
        mlflow_client, model_name=model_name, model_version=model_version
    )
    global_vars["model"] = mlf_model
    global_vars["model_meta"] = mlf_model_meta
    global_vars["aggregator"] = mlf_model.unwrap_python_model().create_aggregator()


def aggregate_parameters(
    parameter_list: list[ParametersRecord],
    global_vars: dict[str, Any],
) -> ParametersRecord:
    aggregator = global_vars["aggregator"]
    agg_parameters = aggregator.aggregate_parameters(parameter_list)
    return agg_parameters


def aggregate_metrics(
    metrics_list: list[MetricsRecord],
    global_vars: dict[str, Any],
) -> dict[str, Any]:
    aggregator = global_vars["aggregator"]
    agg_metrics = aggregator.aggregate_metrics(metrics_list)
    return agg_metrics


def filter_clients(
    messages: list[Message],
) -> list[int]:
    filtered_node_ids = []
    for msg in messages:
        participate = msg.content.configs_records["config"]["participate"]
        if participate:
            print(f"Client {msg.metadata.src_node_id} will participate")
            filtered_node_ids.append(msg.metadata.src_node_id)
        else:
            print(f"Client {msg.metadata.src_node_id} will not participate")

    return filtered_node_ids


def check_success_clients(
    messages: list[Message],
) -> None:
    for msg in messages:
        if msg.has_error():
            print(
                f"Client {msg.metadata.src_node_id} raised error {msg.error.code}: {msg.error.reason}"
            )
        else:
            success = msg.content.configs_records["config"]["success"]
            if not isinstance(success, bool):
                raise TypeError(f"Success field must be a boolean, got {type(success)}")
            if success:
                print(
                    f"Client {msg.metadata.src_node_id} {str(msg.content.configs_records['config']['message'])}"
                )
