import random

from flwr.common import DEFAULT_TTL, MessageType, MetricsRecord, ParametersRecord
from flwr.server import Driver

from fl_server.src.stages import check_success_clients, filter_clients

from common.utils.src.flower_utils.requires import (
    create_filter_clients_recordset,
    create_get_parameters_recordset,
    create_load_data_recordset,
    create_load_model_recordset,
    create_prepare_data_recordset,
    create_set_parameters_recordset,
    create_set_run_recordset,
    create_train_model_recordset,
    create_upload_model_recordset,
)
from common.utils.src.flower_utils.driver import (
    create_messages,
    send_messages,
    wait_messages,
)


def require_filter_clients(driver: Driver, node_ids: list[int]) -> list[int]:
    # Create message content
    recordset = create_filter_clients_recordset()
    # Create messages
    messages = create_messages(
        driver, recordset, MessageType.QUERY, node_ids, "filter_message", DEFAULT_TTL
    )
    # Send messages
    message_ids = send_messages(driver, messages)
    # Wait for results
    all_replies = wait_messages(driver, message_ids)
    # Filter clients
    filtered_node_ids = filter_clients(all_replies)
    return filtered_node_ids


def require_load_data(driver: Driver, node_ids: list[int], use_case: str) -> None:
    # Create message content
    recordset = create_load_data_recordset(use_case)
    # Create messages
    messages = create_messages(
        driver, recordset, MessageType.QUERY, node_ids, "load_data", DEFAULT_TTL
    )
    # Send messages
    message_ids = send_messages(driver, messages)
    # Wait for results
    all_replies = wait_messages(driver, message_ids)
    # Check success
    check_success_clients(all_replies)


def require_prepare_data(driver: Driver, node_ids: list[int]) -> None:
    # Create message content
    recordset = create_prepare_data_recordset()
    # Create messages
    messages = create_messages(
        driver, recordset, MessageType.QUERY, node_ids, "prepare_data", DEFAULT_TTL
    )
    # Send messages
    message_ids = send_messages(driver, messages)
    # Wait for results
    all_replies = wait_messages(driver, message_ids)
    # Check success
    check_success_clients(all_replies)


def require_load_model(
    driver: Driver, node_ids: list[int], model_name: str, model_version: int
) -> None:
    recordset = create_load_model_recordset(model_name, model_version)
    # Create messages
    messages = create_messages(
        driver, recordset, MessageType.QUERY, node_ids, "load_model", DEFAULT_TTL
    )
    # Send messages
    message_ids = send_messages(driver, messages)
    # Wait for results
    all_replies = wait_messages(driver, message_ids)
    # Check success
    check_success_clients(all_replies)


def require_upload_model(
    driver: Driver, node_ids: list[int], parametersrecord: ParametersRecord
) -> None:
    recordset = create_upload_model_recordset(parametersrecord)
    # Pick random client
    node_id = random.choice(node_ids)
    messages = create_messages(
        driver, recordset, MessageType.QUERY, [node_id], "upload_model", DEFAULT_TTL
    )
    # Send messages
    message_ids = send_messages(driver, messages)
    # Wait for results
    all_replies = wait_messages(driver, message_ids)
    # Check success
    check_success_clients(all_replies)


def require_get_parameters_from_one_node(
    driver: Driver, node_ids: list[int]
) -> ParametersRecord:
    recordset = create_get_parameters_recordset()
    # Pick random client
    node_id = random.choice(node_ids)
    messages = create_messages(
        driver, recordset, MessageType.QUERY, [node_id], "get_parameters", DEFAULT_TTL
    )
    # Send messages
    message_ids = send_messages(driver, messages)
    # Wait for results
    all_replies = wait_messages(driver, message_ids)
    msg = all_replies[0]
    # Check success
    check_success_clients(all_replies)
    return msg.content.parameters_records["parameters"]


def require_set_parameters(
    driver: Driver, node_ids: list[int], parametersrecord: ParametersRecord
) -> None:
    recordset = create_set_parameters_recordset(parametersrecord)
    # Create messages
    messages = create_messages(
        driver, recordset, MessageType.QUERY, node_ids, "set_parameters", DEFAULT_TTL
    )
    # Send messages
    message_ids = send_messages(driver, messages)
    # Wait for results
    all_replies = wait_messages(driver, message_ids)
    # Check success
    check_success_clients(all_replies)


def require_set_run_config(
    driver: Driver, node_ids: list[int], experiment_id: int, run_id: int
) -> None:
    recordset = create_set_run_recordset(experiment_id, run_id)
    # Create messages
    messages = create_messages(
        driver, recordset, MessageType.QUERY, node_ids, "set_run_config", DEFAULT_TTL
    )
    # Send messages
    message_ids = send_messages(driver, messages)
    # Wait for results
    all_replies = wait_messages(driver, message_ids)
    # Check success
    check_success_clients(all_replies)


def require_train_model(
    driver: Driver,
    node_ids: list[int],
    parametersrecord: ParametersRecord,
    current_global_iter: int,
) -> list[tuple[ParametersRecord, MetricsRecord]]:
    recordset = create_train_model_recordset(parametersrecord, current_global_iter)
    # Create messages
    messages = create_messages(
        driver, recordset, MessageType.TRAIN, node_ids, "train_model", DEFAULT_TTL
    )
    # Send messages
    message_ids = send_messages(driver, messages)
    # Wait for results
    all_replies = wait_messages(driver, message_ids)
    # Check success
    check_success_clients(all_replies)
    return [
        (
            msg.content.parameters_records["parameters"],
            msg.content.metrics_records["metrics"],
        )
        for msg in all_replies
    ]
