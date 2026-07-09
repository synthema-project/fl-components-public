from random import choice
import structlog
from flwr.common import (
    MessageType,
    MetricsRecord,
    ParametersRecord,
)
from flwr.server import Driver

from fl_server import stages
from fl_server.utils import driver_utils, requires_utils

import interfaces.recordset
logger = structlog.get_logger()


def execution_flow(
    driver,
    node_ids,
    msg_type,
    msg_group,
    recordset_factory=None,
    recordset_factory_args=None,
    per_node_recordset_factory=None,
):
    """Pure transport: build messages, send them, and return the replies.

    This function does not interpret the reply payload. Each caller is
    responsible for validating or extracting its own reply schema:

    * Command messages (``set_run_config``, ``load_model``, ...) reply with a
      ``success`` acknowledgement and should call
      ``requires_utils.check_success_clients(replies)``.
    * Query messages (``filter_clients``, ``get_parameters``, ...) reply with
      data and should interpret it themselves.
    """
    messages = []

    for node_id in node_ids:
        if per_node_recordset_factory is not None:
            recordset = per_node_recordset_factory(node_id)
        else:
            recordset = recordset_factory(*recordset_factory_args)

        messages.extend(
            driver_utils.create_messages(
                driver=driver,
                dst_node_id=[node_id],
                content=recordset,
                message_type=msg_type,
                group_id=msg_group,
            )
        )

    message_ids = driver_utils.send_messages(driver, messages)

    replies = driver_utils.wait_messages(driver, message_ids)

    return replies


def filter_clients(
    driver: Driver,
    node_ids: list[int],
    use_case: str,
    node_dataset_map: dict[str, list[str]],
) -> list[int]:
    # The node -> dataset map is resolved from the catalogue by the REST API at
    # task-creation time and travels inside the task (see schemas.task). The server
    # just broadcasts it; each client self-selects by NODE_NAME.
    all_replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_filter_clients_recordset,
        msg_type=MessageType.QUERY,
        msg_group="filter_message",
        recordset_factory_args=(use_case, node_dataset_map),
    )
    filtered_node_ids = stages.filter_clients(all_replies)
    return filtered_node_ids


def load_data(
    driver,
    node_ids: list[int],
    use_case: str,
    node_dataset_map: dict[str, list[str]],
):
    # Broadcast the full catalogue map; each client selects its own datasets by
    # NODE_NAME. The server cannot map Flower node ids to catalogue node names, so
    # self-selection on the client is the reliable seam.
    replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_load_data_recordset,
        msg_type=MessageType.QUERY,
        msg_group="load_data",
        recordset_factory_args=(use_case, node_dataset_map),
    )
    requires_utils.check_success_clients(replies)

def prepare_data(driver: Driver, node_ids: list[int]) -> None:
    replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_prepare_data_recordset,
        msg_type=MessageType.QUERY,
        msg_group="prepare_data",
    )
    requires_utils.check_success_clients(replies)


def load_model(
    driver: Driver, node_ids: list[int], model_name: str, model_version: int
) -> None:
    replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_load_model_recordset,
        msg_type=MessageType.QUERY,
        msg_group="load_model",
    )
    requires_utils.check_success_clients(replies)


def upload_model(
    driver: Driver, node_ids: list[int], parametersrecord: ParametersRecord
) -> None:
    selected_node = [choice(node_ids)]
    replies = execution_flow(
        driver=driver,
        node_ids=selected_node,
        recordset_factory=interfaces.recordset.create_upload_model_recordset,
        msg_type=MessageType.QUERY,
        msg_group="upload_model",
        recordset_factory_args=(parametersrecord,),
    )
    requires_utils.check_success_clients(replies)


def get_parameters_from_one_node(
    driver: Driver, node_ids: list[int]
) -> ParametersRecord:
    all_replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_get_parameters_recordset,
        msg_type=MessageType.QUERY,
        msg_group="get_parameters",
    )
    # The get_parameters reply carries a success acknowledgement alongside the
    # parameters, so validate it before extracting the payload.
    requires_utils.check_success_clients(all_replies)
    msg = all_replies[0]
    param_record = msg.content.parameters_records["parameters"]
    if not isinstance(param_record, ParametersRecord):
        raise TypeError(f"Expected ParametersRecord, got {type(param_record)}")
    return param_record


def set_parameters(
    driver: Driver, node_ids: list[int], parametersrecord: ParametersRecord
) -> None:
    replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_set_parameters_recordset,
        msg_type=MessageType.QUERY,
        msg_group="set_parameters",
        recordset_factory_args=(parametersrecord,),
    )
    requires_utils.check_success_clients(replies)


def set_run_config(
    driver: Driver,
    node_ids: list[int],
    experiment_id: str,
    run_id: str,
    model_name: str,
    model_version: int,
    dataset_meta_json: str | None = None,
) -> None:
    replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_set_run_cfg_recordset,
        msg_type=MessageType.QUERY,
        msg_group="set_run_config",
        recordset_factory_args=(experiment_id, run_id, model_name, model_version, dataset_meta_json),
    )
    requires_utils.check_success_clients(replies)


def set_dp_config(
    driver: Driver,
    node_ids: list[int],
    dp_mode: str,
    noise_multiplier: int,
    clipping_norm: int
) -> None:
    replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_set_dp_cfg_recordset,
        msg_type=MessageType.QUERY,
        msg_group="set_dp_config",
        recordset_factory_args=(dp_mode, clipping_norm,noise_multiplier),
    )
    requires_utils.check_success_clients(replies)


def train_model(
    driver: Driver,
    node_ids: list[int],
    parametersrecord: ParametersRecord,
    current_global_iter: int,
) -> list[tuple[ParametersRecord, MetricsRecord]]:
    all_replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_train_model_recordset,
        msg_type=MessageType.TRAIN,
        msg_group="train_model",
        recordset_factory_args=(parametersrecord, current_global_iter),
    )
    requires_utils.check_success_clients(all_replies)

    return [
        (
            msg.content.parameters_records["parameters"],
            msg.content.metrics_records["metrics"],
        )
        for msg in all_replies
    ]

def set_SMPC_enabled(
    driver: Driver,
    node_ids: list[int],
    smpc_enabled: bool,
) -> None:
    replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_set_smpc_enabled_recordset,
        msg_type=MessageType.QUERY,
        msg_group="set_smpc_enabled",
        recordset_factory_args=(smpc_enabled,),
    )
    requires_utils.check_success_clients(replies)


def clean_config(
    driver: Driver,
    node_ids: list[int],
) -> None:
    replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_clean_config_recordset,
        msg_type=MessageType.QUERY,
        msg_group="clean_config",
    )
    requires_utils.check_success_clients(replies)

## Require P2P established
def start_P2P_server(driver: Driver,
    node_ids: list[int],
    port: int
) -> None:
# server needs to notify the client to start establishing the P2P network
    all_replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_start_P2P_server_recordset,
        msg_type=MessageType.QUERY,
        msg_group="start_P2P_server",
        recordset_factory_args=(node_ids, port),
    )
    requires_utils.check_success_clients(all_replies)

    return [
        msg.content.configs_records['config']['grpc_port']
        for msg in all_replies
    ]

def discover_peers(
    driver: Driver,
    node_ids: list[int],
    client_port: int,
    max_discovery_time: int = 60
) -> None:
    # returns list of peers discovered by each client node
    all_replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_discover_peers_recordset,
        msg_type=MessageType.QUERY,
        msg_group="discover_peers",
        recordset_factory_args=(node_ids, client_port, max_discovery_time),
    )
    requires_utils.check_success_clients(all_replies)

    return [
        msg.content.configs_records['config']['peers']
        for msg in all_replies
    ]

def connect_peers(
    driver: Driver,
    node_ids: list[int],
    peers: list[list[str]]
) -> None:
    all_replies = []
    for i, node_id in enumerate(node_ids):
        partial_replies = execution_flow(
            driver=driver,
            node_ids=[node_id],
            recordset_factory=interfaces.recordset.create_connect_peers_recordset,
            msg_type=MessageType.QUERY,
            msg_group="connect_peers",
            recordset_factory_args=([node_id], peers[i]),
        )
        all_replies.extend(partial_replies)

    requires_utils.check_success_clients(all_replies)

    return [msg.content.configs_records['config']['peers'] for msg in all_replies]


def evaluate_model(
    driver: Driver,
    node_ids: list[int],
    parametersrecord: ParametersRecord,
) -> list[MetricsRecord]:
    all_replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_evaluate_recordset,
        msg_type=MessageType.EVALUATE,
        msg_group="evaluate_model",
        recordset_factory_args=(parametersrecord,),
    )
    requires_utils.check_success_clients(all_replies)

    return [
        msg.content.metrics_records["metrics"]
        for msg in all_replies
    ]


def generate_synthetic_data(
    driver: Driver,
    node_ids: list[int],
    n_samples: int,
) -> dict[int, str]:
    """Request synthetic data generation from selected nodes.

    Args:
        driver: Flower driver instance.
        node_ids: IDs of the nodes to contact.
        n_samples: Number of synthetic rows to request.

    Returns:
        Mapping of ``node_id -> synthetic CSV string`` for each node.
    """
    all_replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_generate_synthetic_data_recordset,
        msg_type=MessageType.QUERY,
        msg_group="generate_synthetic_data",
        recordset_factory_args=(n_samples,),
    )
    requires_utils.check_success_clients(all_replies)
    return {
        msg.metadata.dst_node_id: msg.content.configs_records["config"]["synthetic_csv"]
        for msg in all_replies
    }


def validate_synthetic_data(
    driver: Driver,
    node_ids: list[int],
    synthetic_csv: str,
) -> dict[int, dict[str, str]]:
    """Broadcast synthetic CSV to all nodes for federated validation.

    Args:
        driver: Flower driver instance.
        node_ids: IDs of the nodes to contact.
        synthetic_csv: CSV string of synthetic data generated on the elected node.

    Returns:
        Mapping of ``node_id -> {"utility": str, "privacy": str}`` (HTML reports).
    """
    all_replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_validate_synthetic_data_recordset,
        msg_type=MessageType.QUERY,
        msg_group="validate_synthetic_data",
        recordset_factory_args=(synthetic_csv,),
    )
    requires_utils.check_success_clients(all_replies)
    return {
        msg.metadata.dst_node_id: {
            "utility": msg.content.configs_records["config"]["utility"],
            "privacy": msg.content.configs_records["config"]["privacy"],
        }
        for msg in all_replies
    }

