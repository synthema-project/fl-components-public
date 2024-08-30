import pytest
from unittest.mock import Mock, patch

import pandas as pd
from mlflow.pyfunc import PyFuncModel
from torch.utils.data import DataLoader
from flwr.common import MetricsRecord

from fl_client import stages
from fl_client.config import DATA_PATH

from fl_models.iris.fl_model import FLModel
from interfaces.pytorch import (
    pytorch_to_parameter_record,
)


@pytest.fixture(scope="session")
def fl_model():
    return FLModel()


@pytest.fixture(scope="session")
def fl_model_wrapper(fl_model):
    wrapper = Mock(spec=PyFuncModel)
    wrapper.unwrap_python_model.return_value = fl_model
    return wrapper


@pytest.fixture(scope="function")
def global_vars_dict(
    fl_model: FLModel,
):
    global_vars = dict()
    global_vars["local_learner"] = fl_model.create_local_learner()
    global_vars["use_case"] = "test_case"
    global_vars["data"] = pd.read_csv(DATA_PATH)

    # Create mock for model_meta
    global_vars["model_meta"] = Mock()
    global_vars["model_meta"].name = "test_model"
    global_vars["model_meta"].description = "test_description"
    global_vars["model_meta"].tags = {"use_case": global_vars["use_case"]}

    # Create mlflow config
    global_vars["mlflow_experiment_id"] = "test_experiment"
    global_vars["mlflow_run_id"] = "test_run"
    yield global_vars


def test_load_data_success(global_vars_dict):
    global_vars = global_vars_dict
    stages.load_data("iris", global_vars)
    assert isinstance(global_vars["data"], pd.DataFrame)


def test_load_data_error(global_vars_dict):
    with pytest.raises(NotImplementedError):
        stages.load_data("unknown", global_vars_dict)


def test_prepare_data(global_vars_dict):
    global_vars = global_vars_dict
    local_learner = global_vars["local_learner"]
    stages.prepare_data(global_vars)
    assert isinstance(local_learner.dataloader, DataLoader)


def test_load_model(fl_model_wrapper):
    # Mocking the global_vars dictionary
    global_vars = {}

    # Load model from mlflow
    with patch(
        "fl_client.stages.mlflow_client.load_model",
        return_value=fl_model_wrapper,
    ):
        stages.load_model(global_vars)

    # Asserting the global variables are set correctly
    assert isinstance(global_vars["model"], PyFuncModel)
    assert global_vars["local_learner"] is not None


def test_upload_model(global_vars_dict):
    # Create the local learner
    global_vars = global_vars_dict
    local_learner = global_vars["local_learner"]
    parameter_records = pytorch_to_parameter_record(local_learner)
    with patch(
        "fl_client.stages.mlflow_client.upload_final_state"
    ) as mock_upload_model:
        stages.upload_model(parameter_records, global_vars)

    mock_upload_model.assert_called_once()


@patch("fl_client.stages.mlflow_client")
def test_set_run_config(mock_mlflow_client):
    mock_mlflow_client.create_child_run = Mock()
    mock_mlflow_client.set_current_config = Mock()
    mock_mlflow_client.set_dataset_signature = Mock()

    config = {
        "experiment_id": "test_experiment",
        "parent_run_id": "test_run",
        "node_name": "test_node",
        "model_name": "test_model",
        "model_version": 1,
        "data_path": "test_data_path",
    }
    stages.set_run_config(**config)

    mock_mlflow_client.create_child_run.assert_called_once_with(
        config["experiment_id"], config["parent_run_id"], config["node_name"]
    )

    mock_mlflow_client.set_current_config.assert_called_once_with(
        config["experiment_id"],
        config["parent_run_id"],
        mock_mlflow_client.create_child_run.return_value,
        config["model_name"],
        config["model_version"],
    )

    mock_mlflow_client.set_dataset_signature.assert_called_once_with(
        config["node_name"],
        config["data_path"],
        mock_mlflow_client.create_child_run.return_value,
    )


def test_train_model_success(global_vars_dict):
    global_vars = global_vars_dict
    with patch.object(global_vars["local_learner"], "train") as mock_train:
        mock_train.return_value = MetricsRecord({"loss": 0.1})
        with patch("fl_client.stages.mlflow_client.log_metrics") as mock_log_metrics:
            metrics = stages.train_model(global_vars, 1)

    assert metrics is not None
    assert metrics["loss"] == 0.1

    mock_log_metrics.assert_called_once_with(metrics, 1)


def test_train_model_error(global_vars_dict):
    global_vars = global_vars_dict
    with patch.object(global_vars["local_learner"], "train") as mock_train:
        mock_train.return_value = "error"
        with pytest.raises(TypeError):
            stages.train_model(global_vars, 1)


def test_evaluate_model_success(global_vars_dict):
    global_vars = global_vars_dict
    with patch.object(global_vars["local_learner"], "evaluate") as mock_evaluate:
        mock_evaluate.return_value = MetricsRecord({"accuracy": 0.9})
        metrics = stages.evaluate_model(global_vars)

    assert metrics is not None
    assert metrics["accuracy"] == 0.9


def test_evaluate_model_error(global_vars_dict):
    global_vars = global_vars_dict
    with patch.object(global_vars["local_learner"], "evaluate") as mock_evaluate:
        mock_evaluate.return_value = "error"
        with pytest.raises(TypeError):
            stages.evaluate_model(global_vars)


def test_get_model_parameters_success(global_vars_dict):
    global_vars = global_vars_dict
    parameters = stages.get_model_parameters(global_vars)

    assert parameters is not None


def test_get_model_parameters_error(global_vars_dict):
    global_vars = global_vars_dict
    with patch.object(
        global_vars["local_learner"], "get_parameters"
    ) as mock_get_parameters:
        mock_get_parameters.return_value = "error"
        with pytest.raises(TypeError):
            stages.get_model_parameters(global_vars)


@patch("fl_client.stages.mlflow_client.clean_current_config")
def test_clean_current_config(mock_clean_current_config):
    stages.clean_current_config()
    mock_clean_current_config.assert_called_once()
