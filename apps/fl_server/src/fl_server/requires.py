from random import choice
import structlog
from flwr.common import (
    DEFAULT_TTL,
    MessageType,
    MetricsRecord,
    ParametersRecord,
)
from flwr.server import Driver

from fl_server import stages
from fl_server.config import DATA_CATALOGUE_TOKEN, DATA_CATALOGUE_URL
from fl_server.data_catalogue import DataCatalogueClient
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
            recordset = recordset_factory(*(recordset_factory_args or ()))

        message = driver.create_message(
            content=recordset,
            message_type=msg_type,
            dst_node_id=node_id,
            group_id=msg_group,
            ttl=DEFAULT_TTL,
        )

        messages.append(message)

    message_ids = driver_utils.send_messages(driver, messages)

    replies = driver_utils.wait_messages(driver, message_ids)

    if check_success:
        requires_utils.check_success_clients(replies)

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


def load_data(
    driver,
    node_ids: list[int],
    use_case: str,
    metadata_documents: tuple[str, str] | None = None,
):

    if not DATA_CATALOGUE_URL or not DATA_CATALOGUE_TOKEN:
        raise RuntimeError("DATA_CATALOGUE_URL and DATA_CATALOGUE_TOKEN must be configured to load federated datasets")

    catalogue_client = DataCatalogueClient(base_url=DATA_CATALOGUE_URL, token=DATA_CATALOGUE_TOKEN)

    node_dataset_map = catalogue_client.get_use_case_datasets(use_case)
    if metadata_documents is None:
        metadata_json, evaluation_config_json = catalogue_client.get_use_case_metadata(use_case)
    else:
        metadata_json, evaluation_config_json = metadata_documents

    def per_node_recordset_factory(node_id: int):
        node_name = f"NODE{node_id}"
        dataset_paths = node_dataset_map.get(node_name, [])

        return interfaces.recordset.create_load_data_recordset(
            use_case,
            dataset_paths,
            metadata_json,
            evaluation_config_json,
        )

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


def generate_synthetic_data(
    driver: Driver,
    node_ids: list[int],
    n_samples: int,
    use_case: str,
    model_name: str,
):
    selected_node = [choice(node_ids)]
    replies = execution_flow(
        driver=driver,
        node_ids=selected_node,
        recordset_factory=interfaces.recordset.create_generate_synthetic_data_recordset,
        msg_type=MessageType.QUERY,
        msg_group="generate_synthetic_data",
        recordset_factory_args=(n_samples,),
        check_success=True,
    )
    config = replies[0].content.configs_records["config"]
    object_uri = config.get("object_uri")
    node_name = config.get("node_name")
    metadata_uri = config.get("metadata_uri")
    evaluation_uri = config.get("evaluation_uri")
    if not all(isinstance(value, str) for value in (object_uri, node_name, metadata_uri, evaluation_uri)):
        raise RuntimeError("Synthetic-data response did not contain complete object and metadata URIs")
    if not DATA_CATALOGUE_URL or not DATA_CATALOGUE_TOKEN:
        raise RuntimeError("DATA_CATALOGUE_TOKEN must be configured to register synthetic data")
    catalogue_client = DataCatalogueClient(base_url=DATA_CATALOGUE_URL, token=DATA_CATALOGUE_TOKEN)
    catalogue_result = catalogue_client.register_synthetic_pool(
        use_case=use_case,
        model_name=model_name,
        node_name=node_name,
        n_samples=n_samples,
        object_uri=object_uri,
        metadata_uri=metadata_uri,
        evaluation_uri=evaluation_uri,
        token=DATA_CATALOGUE_TOKEN,
    )
    pool_id = catalogue_result.get("pool_id")
    if pool_id:
        validation_replies = execution_flow(
            driver=driver,
            node_ids=node_ids,
            recordset_factory=interfaces.recordset.create_validate_synthetic_pool_recordset,
            msg_type=MessageType.QUERY,
            msg_group=f"validate_pool_{pool_id}",
            recordset_factory_args=(pool_id, object_uri, use_case),
            check_success=True,
        )
        for reply in validation_replies:
            payload = reply.content.configs_records["validation"]
            catalogue_client.register_pool_validation(
                pool_id=pool_id,
                node_name=str(payload["node_name"]),
                metrics=dict(payload.get("metrics", {})),
                report_uri=str(payload["report_uri"]),
            )
    return {"object_uri": object_uri, "catalogue": catalogue_result}


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
