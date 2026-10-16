from unittest.mock import MagicMock, patch

import pytest
from flwr.common import (
    MessageType,
    DEFAULT_TTL,
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
    message_ids = ["message-1", "message-2", "message-3"]
    replies = [MagicMock(spec=Message)]

    with (
        patch.object(driver_utils, "send_messages", return_value=message_ids) as send_messages_mock,
        patch.object(driver_utils, "wait_messages", return_value=replies) as wait_messages_mock,
        patch.object(
            requires_utils, "check_success_clients"
        ) as check_success_clients_mock,
    ):
        requires.execution_flow(
            driver=driver,
            node_ids=node_ids,
            recordset_factory=recordset_factory_mock,
            msg_type=MessageType.QUERY,
            msg_group="group",
            recordset_factory_args=("arg1", "arg2"),
            check_success=True,
        )

        assert recordset_factory_mock.call_count == len(node_ids)
        assert driver.create_message.call_count == len(node_ids)
        send_args = send_messages_mock.call_args.args
        assert send_args[0] is driver
        assert len(send_args[1]) == len(node_ids)
        wait_messages_mock.assert_called_once_with(driver, message_ids)
        check_success_clients_mock.assert_called_once_with(replies)


def test_filter_clients(driver, node_ids, use_case):
    with (
        patch.object(requires, "execution_flow") as mock_flow,
        patch.object(requires, "resolve_node_selection") as mock_filter_clients,
    ):
        requires.filter_clients(driver, node_ids, use_case)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_filter_clients_recordset,
        msg_type=MessageType.QUERY,
        msg_group="filter_message",
        recordset_factory_args=(use_case,),
        check_success=False,
    )

    mock_filter_clients.assert_called_once_with(mock_flow.return_value, None)


def test_prepare_data(driver, node_ids):
    with patch.object(requires, "execution_flow") as mock_flow:
        requires.prepare_data(driver, node_ids)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_prepare_data_recordset,
        msg_type=MessageType.QUERY,
        msg_group="prepare_data",
        check_success=True,
    )


def test_load_data(driver, node_ids, use_case):
    with (
        patch.object(requires, "execution_flow") as mock_flow,
        patch.object(requires, "DataCatalogueClient") as client_class,
        patch.object(requires, "DATA_CATALOGUE_URL", "http://catalogue"),
        patch.object(requires, "DATA_CATALOGUE_TOKEN", "token"),
    ):
        client_class.return_value.get_use_case_datasets.return_value = {f"NODE{i}": ["dataset.csv"] for i in node_ids}
        client_class.return_value.get_use_case_metadata.return_value = ("meta", "evaluation")
        requires.load_data(driver, node_ids, use_case, node_names={i: f"NODE{i}" for i in node_ids})

    kwargs = mock_flow.call_args.kwargs
    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        msg_type=MessageType.QUERY,
        msg_group="load_data",
        per_node_recordset_factory=kwargs["per_node_recordset_factory"],
        check_success=True,
    )
    recordset = kwargs["per_node_recordset_factory"](1)
    assert interfaces.recordset.read_load_data_recordset(recordset) == (
        use_case,
        ["dataset.csv"],
        "meta",
        "evaluation",
    )


def test_load_model(driver, node_ids):
    model_name = "model_name"
    model_version = 1

    with patch.object(requires, "execution_flow") as mock_flow:
        requires.load_model(driver, node_ids, model_name, model_version)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_load_model_recordset,
        msg_type=MessageType.QUERY,
        msg_group="load_model",
        check_success=True,
    )


def test_upload_model(driver, node_ids):
    import random

    random.seed(13)
    selected_id = [random.choice(node_ids)]

    random.seed(13)

    pr = ParametersRecord({"param1": MagicMock(Array)})

    with patch.object(requires, "execution_flow") as mock_flow:
        requires.upload_model(driver, node_ids, pr)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=selected_id,
        recordset_factory=interfaces.recordset.create_upload_model_recordset,
        msg_type=MessageType.QUERY,
        msg_group="upload_model",
        recordset_factory_args=(pr,),
        check_success=True,
    )


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
        check_success=True,
    )

    assert param_record == message.content.parameters_records["parameters"]


def test_set_parameters(driver, node_ids):
    pr = ParametersRecord()

    with patch.object(requires, "execution_flow") as mock_flow:
        requires.set_parameters(driver, node_ids, pr)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_set_parameters_recordset,
        msg_type=MessageType.QUERY,
        msg_group="set_parameters",
        recordset_factory_args=(pr,),
        check_success=True,
    )


def test_set_run_config(driver, node_ids):
    model_name = "model_name"
    model_version = 1
    experiment_id = "experiment_id"
    run_id = "run_id"

    with patch.object(requires, "execution_flow") as mock_flow:
        requires.set_run_config(
            driver, node_ids, experiment_id, run_id, model_name, model_version
        )

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_set_run_cfg_recordset,
        msg_type=MessageType.QUERY,
        msg_group="set_run_config",
        recordset_factory_args=(experiment_id, run_id, model_name, model_version),
        check_success=True,
    )


def test_train_model(driver, node_ids):
    pr = ParametersRecord()
    mr = MetricsRecord()
    current_global_iter = 1

    with patch.object(requires, "execution_flow") as mock_flow:
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
        all_replies = requires.train_model(driver, node_ids, pr, current_global_iter, 5)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_train_model_recordset,
        msg_type=MessageType.TRAIN,
        msg_group="train_model",
        recordset_factory_args=(pr, current_global_iter, 5),
        check_success=True,
    )

    assert all_replies == [
        (
            msg.content.parameters_records["parameters"],
            msg.content.metrics_records["metrics"],
        )
        for msg in mock_flow.return_value
    ]


def test_clean_config(driver, node_ids):
    with patch.object(requires, "execution_flow") as mock_flow:
        requires.clean_config(driver, node_ids)

    mock_flow.assert_called_once_with(
        driver=driver,
        node_ids=node_ids,
        recordset_factory=interfaces.recordset.create_clean_config_recordset,
        msg_type=MessageType.QUERY,
        msg_group="clean_config",
        check_success=True,
        recordset_factory_args=("FINISHED",),
        timeout=15,
    )


def test_round_evaluation_protocol_filters_skips_and_preserves_context(driver, node_ids):
    parameters = ParametersRecord()
    def flow(**kwargs):
        assert kwargs['msg_type'] == MessageType.EVALUATE
        assert kwargs['check_success'] is True
        content = kwargs['recordset_factory'](*kwargs['recordset_factory_args'])
        received, iteration, max_round = interfaces.recordset.read_evaluate_model_recordset(content)
        assert received is parameters
        assert (iteration, max_round) == (1, 5)
        return [MagicMock(content=RecordSet(metrics_records={'metrics': metrics}))
                for metrics in [MetricsRecord({'loss': 0.25}), MetricsRecord(), MetricsRecord()]]

    with patch.object(requires, 'execution_flow', side_effect=flow):
        records = requires.evaluate_model(driver, node_ids, parameters, 1, 5)
    assert len(records) == 1
    assert records[0]['loss'] == 0.25


def test_round_evaluation_rejects_missing_client_reply(driver, node_ids):
    with patch.object(requires, 'execution_flow', return_value=[]):
        with pytest.raises(RuntimeError, match='every client reply'):
            requires.evaluate_model(driver, node_ids, ParametersRecord(), 0, 5)


def test_step_zero_collects_then_broadcasts_and_waits_for_every_client(driver, node_ids):
    import json
    from types import SimpleNamespace
    from testing import dataset_model
    from flwr.common import ConfigsRecord
    schema = dataset_model().model_dump(mode='json')
    events = []
    def flow(**kwargs):
        content = kwargs['recordset_factory'](*kwargs['recordset_factory_args'])
        config = content.configs_records['config']
        events.append(config['mode'])
        if config['mode'] == 'compute_step_zero':
            return [SimpleNamespace(content=RecordSet(configs_records={
                'step_zero': ConfigsRecord({'payload_json': json.dumps({
                    'metadata': schema, 'metrics': [{'feature_name': 'Age', 'count': 1.,
                    'sum_val': float(age), 'sum_sq_val': float(age * age)}]})})}))
                    for age in [10, 20, 30]]
        metadata = json.loads(config['metadata_json'])
        assert metadata['metadata']['variables'][0]['mean'] == 20.
        return [object() for _ in node_ids]
    state = {}
    with patch.object(requires, 'execution_flow', side_effect=flow):
        requires.execute_step_zero(driver, node_ids, state)
    assert events == ['compute_step_zero', 'apply_step_zero']
    assert state['step_zero_metadata']['metadata']['variables'][0]['mean'] == 20.


def test_step_zero_missing_reply_stops_before_broadcast(driver, node_ids):
    with patch.object(requires, 'execution_flow', return_value=[]) as flow:
        with pytest.raises(RuntimeError, match='every client reply'):
            requires.execute_step_zero(driver, node_ids, {})
    assert flow.call_count == 1


def test_step_zero_missing_apply_acknowledgement_fails(driver, node_ids):
    import json
    from types import SimpleNamespace
    from flwr.common import ConfigsRecord
    replies = [SimpleNamespace(content=RecordSet(configs_records={
        'step_zero': ConfigsRecord({'payload_json': json.dumps({'metadata': None, 'metrics': []})})}))
        for _ in node_ids]
    with patch.object(requires, 'execution_flow', side_effect=[replies, []]):
        with pytest.raises(RuntimeError, match='every client acknowledgement'):
            requires.execute_step_zero(driver, node_ids, {})


@pytest.mark.parametrize('missing', [False, True])
def test_model_budget_collection_matches_msi_accountant_and_marks_partial(driver, node_ids, missing):
    import json
    from types import SimpleNamespace
    values = [{'epsilon': 1., 'delta': 1e-5}, {'epsilon': 3., 'delta': 1e-6},
              None if missing else {'epsilon': 2., 'delta': 1e-7}]
    replies = [SimpleNamespace(content=RecordSet(configs_records={
        'privacy': ConfigsRecord({'budget_json': json.dumps(value)})})) for value in values]
    with patch.object(requires, 'execution_flow', return_value=replies):
        result = requires.collect_privacy_budgets(driver, node_ids)
    assert result['complete'] is (not missing)
    if missing:
        assert 'model_reported_budget' not in result
        assert result['reported_nodes'] == 2
    else:
        assert result['model_reported_budget'] == {'epsilon': 3., 'delta': 1e-5}
