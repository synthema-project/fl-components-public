from typing import Any
from fl_server.utils.dp_utils import add_gaussian_noise_global_parameters
from flwr.common import Message, ParametersRecord, MetricsRecord
# Use the fl_server-local mlflow client — the one the server actually
# initializes (main.setup_mlflow) and configures (configure_mlflow). Importing
# the shared `interfaces.mlflow_client` here hit a second, never-initialized
# module and raised "expects True, received False" in load_model().
from fl_server.interfaces import mlflow_client


def load_model(
    global_vars: dict[str, Any],
) -> None:
    mlf_model = mlflow_client.load_model()
    global_vars["model"] = mlf_model
    global_vars["aggregator"] = mlf_model.unwrap_python_model().create_aggregator()

def set_SMPC_enabled(
    smpc_enabled: bool,
    global_vars: dict[str, Any],
) -> None:
    global_vars["SMPC_enabled"] = smpc_enabled
    print(f"SMPC enabled set to {smpc_enabled}")

def configure_dp(global_vars: dict[str,Any], dp_config: str, noise_multiplier: int, clipping_norm: int, number_of_clients: int) -> None:
    global_vars["dp_config"]= {
        "dp_mode": dp_config,
        "noise_multiplier": noise_multiplier,
        "clipping_norm": clipping_norm,
        "number_of_clients":number_of_clients
    }

def aggregate_parameters(
    parameter_list: list[ParametersRecord],
    global_vars: dict[str, Any],
) -> ParametersRecord:

    aggregator = global_vars["aggregator"]
    agg_parameters = aggregator.aggregate_parameters(parameter_list)

    if global_vars.get("dp_config"):
        dp_mode = global_vars["dp_config"]["dp_mode"]
        noise_multiplier = global_vars["dp_config"]["noise_multiplier"]
        clipping_norm = global_vars["dp_config"]["clipping_norm"]
        number_of_clients = global_vars["dp_config"]["number_of_clients"]
        if dp_mode == "central_dp":
            add_gaussian_noise_global_parameters(agg_parameters,noise_multiplier,clipping_norm,number_of_clients)

    if not isinstance(agg_parameters, ParametersRecord):
        raise TypeError(
            f"Aggregated parameters must be a ParametersRecord, got {type(agg_parameters)}"
        )
    return agg_parameters


def aggregate_metrics(
    metrics_list: list[MetricsRecord],
    global_vars: dict[str, Any],
) -> MetricsRecord:
    aggregator = global_vars["aggregator"]
    agg_metrics = aggregator.aggregate_metrics(metrics_list)
    if not isinstance(agg_metrics, MetricsRecord):
        raise TypeError(
            f"Aggregated metrics must be a dictionary, got {type(agg_metrics)}"
        )
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
