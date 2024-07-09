from unittest.mock import Mock, patch

from flwr.common import ConfigsRecord, Message, Metadata, ParametersRecord, RecordSet
from mlflow.entities.model_registry import ModelVersion
from fl_client.src.interface import load_data_if, load_model_if, upload_model_if


@patch("fl_client.src.interface.load_data")
def test_load_data_if(load_data_mock):
    global_vars = dict()
    configs_records = {"config": ConfigsRecord({"use_case": "iris"})}
    msg = Message(
        metadata=Mock(spec=Metadata), content=RecordSet(configs_records=configs_records)
    )

    rsp = load_data_if(msg, global_vars)

    assert len(global_vars.keys()) == 0
    assert load_data_mock.called
    assert "config" in rsp.content.configs_records.keys()
    assert rsp.content.configs_records["config"]["success"]


@patch("fl_client.src.interface.load_model")
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
    mlflow_client = Mock()

    rsp = load_model_if(msg, mlflow_client, global_vars)

    assert len(global_vars.keys()) == 0
    assert load_model_mock.called
    assert "config" in rsp.content.configs_records.keys()
    assert rsp.content.configs_records["config"]["success"]


@patch("fl_client.src.interface.upload_model")
def test_upload_model_if(upload_model_mock):
    model_version_mock = Mock(spec=ModelVersion)
    model_version_mock.model_uuid = "model_uuid"
    model_version_mock.run_id = "run_id"
    upload_model_mock.return_value = model_version_mock

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
    mlflow = Mock()
    mlflow_client = Mock()

    rsp = upload_model_if(msg, mlflow, mlflow_client, global_vars)

    assert len(global_vars.keys()) == 0
    assert upload_model_mock.called
    assert "config" in rsp.content.configs_records.keys()
    assert rsp.content.configs_records["config"]["success"]
    assert rsp.content.configs_records["config"]["model_id"] == "model_uuid"
    assert rsp.content.configs_records["config"]["run_id"] == "run_id"
