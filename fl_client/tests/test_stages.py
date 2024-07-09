import pytest
from unittest.mock import Mock, patch, MagicMock


import pandas as pd
from mlflow.pyfunc import PyFuncModel
from fl_client.src.stages import load_data, load_model, prepare_data, upload_model
from fl_client.src.config import DATA_PATH
from common.fl_models.iris.fl_model import FLModel
from torch.utils.data import DataLoader
from common.utils.src.ml_integrations.pytorch import (
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


def test_load_data(global_vars_dict):
    global_vars = global_vars_dict
    load_data("iris", global_vars)
    assert isinstance(global_vars["data"], pd.DataFrame)


def test_prepare_data(global_vars_dict):
    global_vars = global_vars_dict
    local_learner = global_vars["local_learner"]
    prepare_data(global_vars)
    assert isinstance(local_learner.dataloader, DataLoader)


def test_load_model(fl_model_wrapper, global_vars_dict):
    # Run test_model
    model_name, model_version = "iris", 1

    # Mocking the global_vars dictionary
    global_vars = {}

    # Load model from mlflow
    with patch(
        "fl_client.src.stages.load_mlflow_model",
        return_value=(fl_model_wrapper, global_vars_dict),
    ):
        load_model(model_name, model_version, None, global_vars)

    # Asserting the global variables are set correctly
    assert isinstance(global_vars["model"], PyFuncModel)
    assert global_vars["local_learner"] is not None
    assert global_vars["model_meta"] is not None


def test_upload_model(global_vars_dict):
    # Create the local learner
    global_vars = global_vars_dict
    local_learner = global_vars["local_learner"]
    parameter_records = pytorch_to_parameter_record(local_learner)
    with patch("common.utils.src.mlflow_utils.upload_final_state", return_value=Mock()):
        with patch("common.utils.src.mlflow_utils.register_model_metadata"):
            upload_model(parameter_records, MagicMock(), MagicMock(), global_vars)
