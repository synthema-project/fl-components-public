from random import choice
from typing import Callable
import structlog
from flwr.common import (
    DEFAULT_TTL,
    MessageType,
    MetricsRecord,
    ParametersRecord,
    Message,
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
    check_success=True,
    recordset_factory=None,
    recordset_factory_args=None,
    per_node_recordset_factory=None,
):
    messages = []

    for node_id in node_ids:

        # NEW LOGIC: per-node factory has priority
        if per_node_recordset_factory is not None:
            recordset = per_node_recordset_factory(node_id)

        else:
            recordset = recordset_factory(*recordset_factory_args)

        message = driver_utils.create_messages(
            driver=driver,
            node_id=node_id,
            recordset=recordset,
            msg_type=msg_type,
            msg_group=msg_group,
        )

        messages.append(message)

    driver_utils.send_messages(driver, messages)

    replies = driver_utils.wait_messages(driver, messages)

    if check_success:
        driver_utils.check_replies_success(replies)

    return replies


def filter_clients(driver: Driver, node_ids: list[int], use_case: str) -> list[int]:
    all_replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_filter_clients_recordset,
        msg_type=MessageType.QUERY,
        msg_group="filter_message",
        recordset_factory_args=(use_case,),
    )
    filtered_node_ids = stages.filter_clients(all_replies)
    return filtered_node_ids


def load_data(driver, node_ids: list[int], use_case: str):

    catalogue_client = DataCatalogueClient(
        base_url=DATA_CATALOGUE_URL
    )

    node_dataset_map = catalogue_client.get_use_case_datasets(use_case)

    def per_node_recordset_factory(node_id: int):
        node_name = f"NODE{node_id}"
        dataset_paths = node_dataset_map.get(node_name, [])

        return create_load_data_recordset(use_case, dataset_paths)

    execution_flow(
        driver=driver,
        node_ids=node_ids,
        msg_type=MessageType.QUERY,
        msg_group="load_data",
        per_node_recordset_factory=per_node_recordset_factory,
        check_success=True,
    )

def prepare_data(driver: Driver, node_ids: list[int]) -> None:
    execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_prepare_data_recordset,
        msg_type=MessageType.QUERY,
        msg_group="prepare_data",
        check_success=True,
    )


def load_model(
    driver: Driver, node_ids: list[int], model_name: str, model_version: int
) -> None:
    execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_load_model_recordset,
        msg_type=MessageType.QUERY,
        msg_group="load_model",
        check_success=True,
    )


def upload_model(
    driver: Driver, node_ids: list[int], parametersrecord: ParametersRecord
) -> None:
    selected_node = [choice(node_ids)]
    execution_flow(
        driver=driver,
        node_ids=selected_node,
        recordset_factory=interfaces.recordset.create_upload_model_recordset,
        msg_type=MessageType.QUERY,
        msg_group="upload_model",
        recordset_factory_args=(parametersrecord,),
        check_success=True,
    )


def get_parameters_from_one_node(
    driver: Driver, node_ids: list[int]
) -> ParametersRecord:
    all_replies = execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_get_parameters_recordset,
        msg_type=MessageType.QUERY,
        msg_group="get_parameters",
        check_success=True,
    )
    msg = all_replies[0]
    # Check success
    requires_utils.check_success_clients(all_replies)
    param_record = msg.content.parameters_records["parameters"]
    if not isinstance(param_record, ParametersRecord):
        raise TypeError(f"Expected ParametersRecord, got {type(param_record)}")
    return param_record


def set_parameters(
    driver: Driver, node_ids: list[int], parametersrecord: ParametersRecord
) -> None:
    execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_set_parameters_recordset,
        msg_type=MessageType.QUERY,
        msg_group="set_parameters",
        recordset_factory_args=(parametersrecord,),
        check_success=True,
    )


def set_run_config(
    driver: Driver,
    node_ids: list[int],
    experiment_id: str,
    run_id: str,
    model_name: str,
    model_version: int,
) -> None:
    execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_set_run_cfg_recordset,
        msg_type=MessageType.QUERY,
        msg_group="set_run_config",
        recordset_factory_args=(experiment_id, run_id, model_name, model_version),
        check_success=True,
    )


def set_dp_config(
    driver: Driver,
    node_ids: list[int],
    dp_mode: str,
    noise_multiplier: int,
    clipping_norm: int
) -> None:
    execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_set_dp_cfg_recordset,
        msg_type=MessageType.QUERY,
        msg_group="set_dp_config",
        recordset_factory_args=(dp_mode, clipping_norm,noise_multiplier),
        check_success=True,
    )


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
        check_success=True,
    )

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
    execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_set_smpc_enabled_recordset,
        msg_type=MessageType.QUERY,
        msg_group="set_smpc_enabled",
        recordset_factory_args=(smpc_enabled,),
        check_success=True,
    )


def clean_config(
    driver: Driver,
    node_ids: list[int],
) -> None:
    execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_clean_config_recordset,
        msg_type=MessageType.QUERY,
        msg_group="clean_config",
        check_success=True,
    )

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
        check_success=True,
    )

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
        check_success=True,
    )

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

    return [msg.content.configs_records['config']['peers'] for msg in all_replies]
