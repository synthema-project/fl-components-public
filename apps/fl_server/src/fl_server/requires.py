from typing import Callable

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


def execution_flow(
    driver: Driver,
    node_ids: list[int],
    recordset_factory: Callable,
    msg_type: str,
    msg_group: str,
    recordset_factory_args: tuple = (),
    check_success: bool = False,
) -> list[Message]:
    recordset = recordset_factory(*recordset_factory_args)
    messages = driver_utils.create_messages(
        driver, recordset, msg_type, node_ids, msg_group, DEFAULT_TTL
    )
    message_ids = driver_utils.send_messages(driver, messages)
    all_replies = driver_utils.wait_messages(driver, message_ids)
    if check_success:
        requires_utils.check_success_clients(all_replies)
    return all_replies


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


def load_data(driver: Driver, node_ids: list[int], use_case: str) -> None:
    execution_flow(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_load_data_recordset,
        msg_type=MessageType.QUERY,
        msg_group="load_data",
        recordset_factory_args=(use_case,),
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
    execution_flow(
        driver=driver,
        node_ids=node_ids,
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
