import pytest
from unittest.mock import Mock, patch


from mlflow.pyfunc import PyFuncModel
from fl_server.stages import load_model
from fl_models.iris.fl_model import FLModel


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
    global_vars["aggregator"] = fl_model.create_aggregator()
    global_vars["use_case"] = "test_case"

    # Create mock for model_meta
    global_vars["model_meta"] = Mock()
    global_vars["model_meta"].name = "test_model"
    global_vars["model_meta"].description = "test_description"
    global_vars["model_meta"].tags = {"use_case": global_vars["use_case"]}

    # Create mlflow config
    global_vars["mlflow_experiment_id"] = "test_experiment"
    global_vars["mlflow_run_id"] = "test_run"
    yield global_vars


def test_load_model(fl_model_wrapper, global_vars_dict):
    # Mocking the global_vars dictionary
    global_vars = {}

    # Load model from mlflow
    with patch(
        "fl_server.stages.mlflow_client.load_model",
        return_value=fl_model_wrapper,
    ):
        load_model(global_vars)

    # Asserting the global variables are set correctly
    assert isinstance(global_vars["model"], PyFuncModel)
    assert global_vars["aggregator"] is not None
    assert global_vars["model"] is not None
