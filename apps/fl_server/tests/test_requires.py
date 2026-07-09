from unittest.mock import MagicMock, patch

import pytest
from flwr.common import (
    MessageType,
    Message,
    RecordSet,
    ParametersRecord,
    Array,
    ConfigsRecord,
    MetricsRecord,
)
from flwr.server import Driver

from fl_server import requires
from fl_server.utils import requires_utils, driver_utils

import interfaces.recordset


@pytest.fixture
def driver():
    return MagicMock(spec=Driver)


@pytest.fixture
def node_ids():
    return [1, 2, 3]


@pytest.fixture
def use_case():
    return "use_case"


def test_execution_flow(driver, node_ids):
    rs = RecordSet()
    recordset_factory_mock = MagicMock(return_value=rs)

    with (
        patch.object(
            driver_utils, "create_messages", return_value=MagicMock(spec=list[Message])
        ) as create_messages_mock,
        patch.object(
            driver_utils, "send_messages", return_value=MagicMock(spec=list[str])
        ) as send_messages_mock,
        patch.object(
            driver_utils, "wait_messages", return_value=MagicMock(spec=list[Message])
        ) as wait_messages_mock,
        patch.object(
            requires_utils, "check_success_clients"
        ) as check_success_clients_mock,
    ):
        replies = requires.execution_flow(
            driver=driver,
            node_ids=node_ids,
            recordset_factory=recordset_factory_mock,
            msg_type=MessageType.QUERY,
            msg_group="group",
            recordset_factory_args=("arg1", "arg2"),
        )

        assert recordset_factory_mock.call_count == len(node_ids)
        assert create_messages_mock.call_count == len(node_ids)
        send_messages_mock.assert_called_once()
        wait_messages_mock.assert_called_once()
        # execution_flow is pure transport: it returns the replies verbatim and
        # does not interpret them (no success validation).
        assert replies == wait_messages_mock.return_value
        check_success_clients_mock.assert_not_called()



def test_filter_clients(driver, node_ids, use_case):
    # The node -> dataset map arrives with the task; filter_clients broadcasts it
    # and does not query the catalogue itself.
    node_dataset_map = {"HULAFE": ["synthema-catalogue/aml/hulafe.csv"]}
    with (
        patch.object(requires, "execution_flow") as mock_flow,
        patch.object(requires.stages, "filter_clients") as mock_filter_clients,
    ):
        requires.filter_clients(driver, node_ids, use_case, node_dataset_map)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_filter_clients_recordset,
        msg_type=MessageType.QUERY,
        msg_group="filter_message",
        recordset_factory_args=(use_case, node_dataset_map),
    )

    mock_filter_clients.assert_called_once_with(mock_flow.return_value)


def test_prepare_data(driver, node_ids):
    with (
        patch.object(requires, "execution_flow") as mock_flow,
        patch.object(requires.requires_utils, "check_success_clients") as mock_check,
    ):
        requires.prepare_data(driver, node_ids)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_prepare_data_recordset,
        msg_type=MessageType.QUERY,
        msg_group="prepare_data",
    )
    mock_check.assert_called_once_with(mock_flow.return_value)


def test_load_data(driver, node_ids, use_case):
    node_dataset_map = {"HULAFE": ["synthema-catalogue/aml/hulafe.csv"]}
    with (
        patch.object(requires, "execution_flow") as mock_flow,
        patch.object(requires.requires_utils, "check_success_clients") as mock_check,
    ):
        requires.load_data(driver, node_ids, use_case, node_dataset_map)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_load_data_recordset,
        msg_type=MessageType.QUERY,
        msg_group="load_data",
        recordset_factory_args=(use_case, node_dataset_map),
    )
    mock_check.assert_called_once_with(mock_flow.return_value)



def test_load_model(driver, node_ids):
    model_name = "model_name"
    model_version = 1

    with (
        patch.object(requires, "execution_flow") as mock_flow,
        patch.object(requires.requires_utils, "check_success_clients") as mock_check,
    ):
        requires.load_model(driver, node_ids, model_name, model_version)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_load_model_recordset,
        msg_type=MessageType.QUERY,
        msg_group="load_model",
    )
    mock_check.assert_called_once_with(mock_flow.return_value)


def test_upload_model(driver, node_ids):
    import random

    random.seed(13)
    selected_id = [random.choice(node_ids)]

    random.seed(13)

    pr = ParametersRecord({"param1": MagicMock(Array)})

    with (
        patch.object(requires, "execution_flow") as mock_flow,
        patch.object(requires.requires_utils, "check_success_clients") as mock_check,
    ):
        requires.upload_model(driver, node_ids, pr)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=selected_id,
        recordset_factory=interfaces.recordset.create_upload_model_recordset,
        msg_type=MessageType.QUERY,
        msg_group="upload_model",
        recordset_factory_args=(pr,),
    )
    mock_check.assert_called_once_with(mock_flow.return_value)


def test_get_parameters_from_one_node(driver, node_ids):
    message = Message(
        metadata=MagicMock(),
        content=RecordSet(
            parameters_records={"parameters": ParametersRecord()},
            configs_records={
                "config": ConfigsRecord({"success": True, "message": "ok"})
            },
        ),
    )

    with patch.object(requires, "execution_flow") as mock_flow:
        mock_flow.return_value = [message]
        param_record = requires.get_parameters_from_one_node(driver, node_ids)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_get_parameters_recordset,
        msg_type=MessageType.QUERY,
        msg_group="get_parameters",
    )

    assert param_record == message.content.parameters_records["parameters"]


def test_set_parameters(driver, node_ids):
    pr = ParametersRecord()

    with (
        patch.object(requires, "execution_flow") as mock_flow,
        patch.object(requires.requires_utils, "check_success_clients") as mock_check,
    ):
        requires.set_parameters(driver, node_ids, pr)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_set_parameters_recordset,
        msg_type=MessageType.QUERY,
        msg_group="set_parameters",
        recordset_factory_args=(pr,),
    )
    mock_check.assert_called_once_with(mock_flow.return_value)


def test_set_run_config(driver, node_ids):
    model_name = "model_name"
    model_version = 1
    experiment_id = "experiment_id"
    run_id = "run_id"

    with (
        patch.object(requires, "execution_flow") as mock_flow,
        patch.object(requires.requires_utils, "check_success_clients") as mock_check,
    ):
        requires.set_run_config(
            driver, node_ids, experiment_id, run_id, model_name, model_version
        )

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_set_run_cfg_recordset,
        msg_type=MessageType.QUERY,
        msg_group="set_run_config",
        recordset_factory_args=(experiment_id, run_id, model_name, model_version, None),
    )
    mock_check.assert_called_once_with(mock_flow.return_value)


def test_train_model(driver, node_ids):
    pr = ParametersRecord()
    mr = MetricsRecord()
    current_global_iter = 1

    with (
        patch.object(requires, "execution_flow") as mock_flow,
        patch.object(requires.requires_utils, "check_success_clients") as mock_check,
    ):
        mock_flow.return_value = [
            Message(
                MagicMock(),
                RecordSet(
                    parameters_records={"parameters": pr},
                    metrics_records={"metrics": mr},
                ),
            )
            for _ in range(3)
        ]
        all_replies = requires.train_model(driver, node_ids, pr, current_global_iter)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_train_model_recordset,
        msg_type=MessageType.TRAIN,
        msg_group="train_model",
        recordset_factory_args=(pr, current_global_iter),
    )
    mock_check.assert_called_once_with(mock_flow.return_value)

    assert all_replies == [
        (
            msg.content.parameters_records["parameters"],
            msg.content.metrics_records["metrics"],
        )
        for msg in mock_flow.return_value
    ]


def test_clean_config(driver, node_ids):
    with (
        patch.object(requires, "execution_flow") as mock_flow,
        patch.object(requires.requires_utils, "check_success_clients") as mock_check,
    ):
        requires.clean_config(driver, node_ids)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_clean_config_recordset,
        msg_type=MessageType.QUERY,
        msg_group="clean_config",
    )
    mock_check.assert_called_once_with(mock_flow.return_value)


def test_evaluate_model(driver, node_ids):
    pr = ParametersRecord()
    mr = MetricsRecord()

    with (
        patch.object(requires, "execution_flow") as mock_flow,
        patch.object(requires.requires_utils, "check_success_clients") as mock_check,
    ):
        mock_flow.return_value = [
            Message(
                MagicMock(),
                RecordSet(
                    metrics_records={"metrics": mr},
                ),
            )
            for _ in range(3)
        ]
        all_replies = requires.evaluate_model(driver, node_ids, pr)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_evaluate_recordset,
        msg_type=MessageType.EVALUATE,
        msg_group="evaluate_model",
        recordset_factory_args=(pr,),
    )
    mock_check.assert_called_once_with(mock_flow.return_value)

    assert all_replies == [
        msg.content.metrics_records["metrics"]
        for msg in mock_flow.return_value
    ]

