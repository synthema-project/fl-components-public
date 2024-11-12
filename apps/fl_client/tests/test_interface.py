from unittest.mock import Mock, patch

import pytest
from flwr.common import (
    ConfigsRecord,
    Message,
    Metadata,
    ParametersRecord,
    RecordSet,
    MetricsRecord,
)

from fl_client import interface


@patch("fl_client.interface.stages.load_data")
def test_load_data_if_success(load_data_mock):
    global_vars = dict()
    configs_records = {"config": ConfigsRecord({"use_case": "iris"})}
    msg = Message(
        metadata=Mock(spec=Metadata), content=RecordSet(configs_records=configs_records)
    )

    rsp = interface.load_data_if(msg, global_vars)

    assert len(global_vars.keys()) == 0
    assert load_data_mock.called
    assert "config" in rsp.content.configs_records.keys()
    assert rsp.content.configs_records["config"]["success"]


@patch("fl_client.interface.stages.load_data")
def test_load_data_if_error(load_data_mock):
    global_vars = dict()
    configs_records = {"config": ConfigsRecord({"use_case": 1})}
    msg = Message(
        metadata=Mock(spec=Metadata), content=RecordSet(configs_records=configs_records)
    )

    with pytest.raises(TypeError):
        interface.load_data_if(msg, global_vars)


@patch("fl_client.interface.stages.prepare_data")
def test_prepare_data_if(prepare_data_mock):
    global_vars = dict()
    msg = Message(metadata=Mock(spec=Metadata), content=RecordSet())

    rsp = interface.prepare_data_if(msg, global_vars)

    assert len(global_vars.keys()) == 0
    assert prepare_data_mock.called
    assert "success" in rsp.content.configs_records["config"].keys()
    assert rsp.content.configs_records["config"]["success"]


@patch("fl_client.interface.stages.load_model")
def test_load_model_if(load_model_mock):
    global_vars = dict()
    configs_records = {
        "config": ConfigsRecord(
            {"model_name": "test_model", "model_version": 1, "use_case": "iris"}
        )
    }
    msg = Message(
        metadata=Mock(spec=Metadata), content=RecordSet(configs_records=configs_records)
    )

    rsp = interface.load_model_if(msg, global_vars)

    assert len(global_vars.keys()) == 0
    assert load_model_mock.called
    assert "config" in rsp.content.configs_records.keys()
    assert rsp.content.configs_records["config"]["success"]


def test_upload_model_if():
    model_info = {
        "model_id": "model_uuid",
        "run_id": "run_id",
    }

    global_vars = dict()
    configs_records = {
        "config": ConfigsRecord({"model_name": "test_model", "model_version": "1"})
    }
    msg = Message(
        metadata=Mock(spec=Metadata),
        content=RecordSet(
            configs_records=configs_records,
            parameters_records={"parameters": Mock(ParametersRecord)},
        ),
    )

    with patch(
        "fl_client.interface.stages.upload_model", return_value=model_info
    ) as mock_upload_model:
        rsp = interface.upload_model_if(msg, global_vars)

    assert len(global_vars.keys()) == 0
    assert mock_upload_model.called
    assert "config" in rsp.content.configs_records.keys()
    assert rsp.content.configs_records["config"]["success"]
    assert rsp.content.configs_records["config"]["model_id"] == "model_uuid"
    assert rsp.content.configs_records["config"]["run_id"] == "run_id"


@patch("fl_client.interface.stages.train_model")
@patch("fl_client.interface.stages.get_model_parameters")
@patch("fl_client.interface.stages.set_model_parameters")
def test_train_model_if_success(
    set_model_parameters_mock, get_model_parameters_mock, train_model_mock
):
    get_model_parameters_mock.return_value = Mock(ParametersRecord)
    train_model_mock.return_value = Mock(MetricsRecord)

    global_vars = dict()
    configs_records = {
        "config": ConfigsRecord({"current_global_iter": 1}),
    }
    msg = Message(
        metadata=Mock(spec=Metadata),
        content=RecordSet(
            configs_records=configs_records,
            parameters_records={"parameters": Mock(ParametersRecord)},
        ),
    )

    rsp = interface.train_model_if(msg, global_vars)

    assert len(global_vars.keys()) == 0
    assert set_model_parameters_mock.called
    assert get_model_parameters_mock.called
    assert train_model_mock.called
    assert "parameters" in rsp.content.parameters_records.keys()
    assert "metrics" in rsp.content.metrics_records.keys()


def test_train_model_if_error():
    global_vars = dict()
    configs_records = {
        "config": ConfigsRecord({"current_global_iter": "1"}),
    }
    msg = Message(
        metadata=Mock(spec=Metadata),
        content=RecordSet(
            configs_records=configs_records,
            parameters_records={"parameters": Mock(ParametersRecord)},
        ),
    )

    with pytest.raises(TypeError):
        interface.train_model_if(msg, global_vars)


@patch("fl_client.interface.stages.get_model_parameters")
def test_get_parameters_if(get_model_parameters_mock):
    get_model_parameters_mock.return_value = Mock(ParametersRecord)

    global_vars = dict()
    msg = Message(
        metadata=Mock(spec=Metadata),
        content=RecordSet(),
    )

    rsp = interface.get_parameters_if(msg, global_vars)

    assert len(global_vars.keys()) == 0
    assert get_model_parameters_mock.called
    assert "parameters" in rsp.content.parameters_records.keys()


@patch("fl_client.interface.stages.set_model_parameters")
def test_set_parameters_if(set_model_parameters_mock):
    global_vars = dict()
    msg = Message(
        metadata=Mock(spec=Metadata),
        content=RecordSet(
            parameters_records={"parameters": Mock(ParametersRecord)},
        ),
    )

    rsp = interface.set_parameters_if(msg, global_vars)

    assert len(global_vars.keys()) == 0
    assert set_model_parameters_mock.called
    assert "success" in rsp.content.configs_records["config"].keys()
    assert rsp.content.configs_records["config"]["success"]


@patch("fl_client.interface.stages.set_run_config")
def test_set_run_config_if_success(set_run_config_mock):
    global_vars = dict(
        node_name="node_name",
        data_path="/path",
    )
    configs_records = {
        "config": ConfigsRecord(
            {
                "experiment_id": "exp_id",
                "run_id": "run_id",
                "model_name": "model_name",
                "model_version": 1,
            }
        ),
    }
    msg = Message(
        metadata=Mock(spec=Metadata),
        content=RecordSet(configs_records=configs_records),
    )

    rsp = interface.set_run_config_if(msg, global_vars)

    assert len(global_vars.keys()) == 2
    assert set_run_config_mock.called
    assert "success" in rsp.content.configs_records["config"].keys()
    assert rsp.content.configs_records["config"]["success"]


def test_set_run_config_if_error():
    global_vars = dict(
        node_name="node_name",
        data_path="/path",
    )
    configs_records = {
        "config": ConfigsRecord(
            {
                "experiment_id": 1,
                "run_id": "run_id",
                "model_name": "model_name",
                "model_version": 1,
            }
        ),
    }
    msg = Message(
        metadata=Mock(spec=Metadata),
        content=RecordSet(configs_records=configs_records),
    )

    with pytest.raises(TypeError):
        interface.set_run_config_if(msg, global_vars)


@patch("fl_client.interface.stages.clean_current_config")
def test_clean_config_if(clean_current_config_mock):
    global_vars = dict()
    msg = Message(
        metadata=Mock(spec=Metadata),
        content=RecordSet(),
    )

    rsp = interface.clean_config_if(msg)

    assert len(global_vars.keys()) == 0
    assert clean_current_config_mock.called
    assert "success" in rsp.content.configs_records["config"].keys()
    assert rsp.content.configs_records["config"]["success"]


def test_filter_clients_if_success():
    global_vars = {"use_case": "test"}
    configs_records = {
        "config": ConfigsRecord({"use_case": "test"}),
    }
    msg = Message(
        metadata=Mock(spec=Metadata),
        content=RecordSet(configs_records=configs_records),
    )

    rsp = interface.filter_clients_if(msg, global_vars)

    assert len(global_vars.keys()) == 1
    assert "participate" in rsp.content.configs_records["config"].keys()
    assert rsp.content.configs_records["config"]["participate"]

    # change use_case
    global_vars["use_case"] = "test2"

    rsp = interface.filter_clients_if(msg, global_vars)

    assert not rsp.content.configs_records["config"]["participate"]


def test_filter_clients_if_error():
    global_vars = {"use_case": "test"}
    configs_records = {
        "config": ConfigsRecord({"use_case": 1}),
    }
    msg = Message(
        metadata=Mock(spec=Metadata),
        content=RecordSet(configs_records=configs_records),
    )

    with pytest.raises(TypeError):
        interface.filter_clients_if(msg, global_vars)
