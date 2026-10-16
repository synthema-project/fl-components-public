from unittest.mock import MagicMock, Mock, patch

import pytest

from mlflow.pyfunc import PyFuncModel
from flwr.common import ParametersRecord, MetricsRecord, Message

from fl_server import stages
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


def test_load_model(fl_model_wrapper):
    # Mocking the global_vars dictionary
    global_vars = {}

    # Load model from mlflow
    with patch(
        "fl_server.stages.mlflow_client.load_model",
        return_value=fl_model_wrapper,
    ):
        stages.load_model(global_vars)

    # Asserting the global variables are set correctly
    assert isinstance(global_vars["model"], PyFuncModel)
    assert global_vars["aggregator"] is not None
    assert global_vars["model"] is not None


def test_aggregate_paramemters(global_vars_dict):
    mock_parameters = MagicMock(spec=ParametersRecord)
    parameter_list = [mock_parameters, mock_parameters]

    aggregator = global_vars_dict["aggregator"]
    with patch.object(aggregator, "aggregate_parameters") as mock_agg_params:
        mock_agg_params.return_value = mock_parameters

        result = stages.aggregate_parameters(parameter_list, global_vars_dict)

    assert isinstance(result, ParametersRecord)
    mock_agg_params.assert_called_once_with(parameter_list)


def test_aggregate_metrics(global_vars_dict):
    mock_metrics = MagicMock(MetricsRecord)
    metrics_list = [mock_metrics, mock_metrics]

    aggregator = global_vars_dict["aggregator"]
    with patch.object(aggregator, "aggregate_metrics") as mock_agg_metrics:
        mock_agg_metrics.return_value = mock_metrics

        result = stages.aggregate_metrics(metrics_list, global_vars_dict)

    assert isinstance(result, MetricsRecord)
    mock_agg_metrics.assert_called_once_with(metrics_list)


def test_filter_clients():
    messages_mock = [MagicMock(Message) for _ in range(3)]
    messages_mock[0].metadata.src_node_id = 1
    messages_mock[0].content.configs_records = {"config": {"participate": True}}
    messages_mock[1].metadata.src_node_id = 2
    messages_mock[1].content.configs_records = {"config": {"participate": False}}
    messages_mock[2].metadata.src_node_id = 3
    messages_mock[2].content.configs_records = {"config": {"participate": True}}

    filtered_node_ids = stages.filter_clients(messages_mock)

    assert filtered_node_ids == [1, 3]


def test_local_dp_preserves_custom_aggregator_and_adds_no_platform_noise():
    from types import SimpleNamespace
    first, second, custom_result = ParametersRecord(), ParametersRecord(), ParametersRecord()
    aggregate = Mock(return_value=custom_result)
    state = {'aggregator': SimpleNamespace(aggregate_parameters=aggregate),
             'SMPC_enabled': False, 'dp_config': {'dp_mode': 'local_dp',
             'noise_multiplier': 1, 'clipping_norm': 1, 'number_of_clients': 2}}
    with patch.object(stages, 'add_gaussian_noise_global_parameters') as noise:
        assert stages.aggregate_parameters([first, second], state) is custom_result
    aggregate.assert_called_once_with([first, second])
    noise.assert_not_called()


def test_existing_central_dp_runs_after_custom_aggregator_and_uses_noise_result():
    from types import SimpleNamespace
    inputs = [ParametersRecord(), ParametersRecord()]
    aggregate_result, noisy_result = ParametersRecord(), ParametersRecord()
    aggregate = Mock(return_value=aggregate_result)
    state = {'aggregator': SimpleNamespace(aggregate_parameters=aggregate),
             'dp_config': {'dp_mode': 'central_dp', 'noise_multiplier': 2,
                           'clipping_norm': 3, 'number_of_clients': 2}}
    with patch.object(stages, 'add_gaussian_noise_global_parameters', return_value=noisy_result) as noise:
        assert stages.aggregate_parameters(inputs, state) is noisy_result
    aggregate.assert_called_once_with(inputs)
    noise.assert_called_once_with(aggregate_result, 2, 3, 2)


def test_partner_gaussian_noise_formula_is_preserved():
    import numpy as np
    from collections import OrderedDict
    from fl_models.iris.utils import Utils
    from fl_server.utils import dp_utils
    parameters = Utils.dict_to_parameter_record(OrderedDict(weight=np.array([2.], dtype=np.float32)))
    with patch.object(dp_utils.np.random, 'normal', return_value=np.array([.25])) as gaussian:
        result = dp_utils.add_gaussian_noise_global_parameters(parameters, 2, 3, 4)
    gaussian.assert_called_once_with(0, 1.5, (1,))
    np.testing.assert_allclose(Utils.parameters_to_dict(result)['weight'], [2.25])
