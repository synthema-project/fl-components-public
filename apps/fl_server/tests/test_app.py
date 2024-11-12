from unittest.mock import MagicMock, patch

import pytest

from flwr.common import ParametersRecord, MetricsRecord

from fl_server.app import _mlflow_config, _model_and_data, _training_loop, get_serverapp
from fl_server import config

from schemas.task import Task


@pytest.fixture
def task():
    return Task(
        user_id="user_id",
        use_case="use_case",
        model_name="model_name",
        model_version=1,
        num_global_iterations=2,
        id=1,
        status="pending",
        created_at="2021-08-01T00:00:00",
        run_name="run_name",
        experiment_name="experiment_name",
    )


@pytest.fixture
def node_ids():
    return [1, 2, 3]


@pytest.fixture
def driver(node_ids):
    mock = MagicMock()
    mock.get_node_ids.return_value = node_ids
    return mock


@pytest.fixture
def parameters():
    return ParametersRecord()


@pytest.fixture
def metrics():
    return MetricsRecord()


@pytest.fixture
def config_dict():
    return {
        "aggregator": "agg",
        "model": "model",
        "model_meta": "model_meta",
        "use_case": "use_case",
    }


def test_mlflow_config(task, driver, node_ids):
    with (
        patch("fl_server.app.mlflow_utils.create_mlflow_runs") as mock_create_runs,
        patch("fl_server.app.mlflow_client.set_current_config") as mock_set_config,
        patch("fl_server.app.requires.set_run_config") as mock_set_run_config,
    ):
        mock_create_runs.return_value = (
            "experiment_id",
            "parent_run_id",
            "child_run_id",
        )

        _mlflow_config(task, driver, node_ids)

        mock_create_runs.assert_called_once_with(task.run_name, task.experiment_name)
        mock_set_config.assert_called_once_with(
            "experiment_id",
            "parent_run_id",
            "child_run_id",
            task.model_name,
            task.model_version,
        )
        mock_set_run_config.assert_called_once_with(
            driver,
            node_ids,
            "experiment_id",
            "parent_run_id",
            task.model_name,
            task.model_version,
        )


def test_model_and_data(task, driver, node_ids, config_dict):
    with (
        patch("fl_server.app.stages.load_model") as mock_stages_load_model,
        patch("fl_server.app.requires.load_data") as mock_requires_load_data,
        patch("fl_server.app.requires.load_model") as mock_requires_load_model,
        patch("fl_server.app.requires.prepare_data") as mock_requires_prepare_data,
        patch.dict(config.global_vars, config_dict) as mock_global_vars,
    ):
        _model_and_data(task, driver, node_ids)

        mock_stages_load_model.assert_called_once_with(mock_global_vars)
        mock_requires_load_data.assert_called_once_with(driver, node_ids, task.use_case)
        mock_requires_load_model.assert_called_once_with(
            driver, node_ids, task.model_name, task.model_version
        )
        mock_requires_prepare_data.assert_called_once_with(driver, node_ids)


def test_training_loop(driver, node_ids, parameters, metrics, config_dict):
    n_global_iter = 10

    with (
        patch("fl_server.app.requires.train_model") as mock_requires_train_model,
        patch(
            "fl_server.app.stages.aggregate_parameters"
        ) as mock_stages_aggregate_parameters,
        patch(
            "fl_server.app.stages.aggregate_metrics"
        ) as mock_stages_aggregate_metrics,
        patch(
            "fl_server.app.mlflow_client.log_metrics"
        ) as mock_mlflow_client_log_metrics,
        patch("fl_server.app.requires.set_parameters") as mock_requires_set_parameters,
        patch.dict(config.global_vars, config_dict),
    ):
        mock_requires_train_model.return_value = [
            (parameters, metrics) for _ in range(n_global_iter)
        ]
        mock_stages_aggregate_parameters.return_value = parameters
        mock_stages_aggregate_metrics.return_value = metrics

        _training_loop(driver, node_ids, parameters, n_global_iter)

        mock_stages_aggregate_parameters.assert_called_with(
            [parameters] * n_global_iter, config_dict
        )
        mock_stages_aggregate_metrics.assert_called_with(
            [metrics] * n_global_iter, config_dict
        )
        mock_mlflow_client_log_metrics.assert_called_with(
            metrics, step=n_global_iter - 1
        )
        mock_requires_set_parameters.assert_called_with(driver, node_ids, parameters)

        assert mock_requires_train_model.call_count == n_global_iter
        assert mock_stages_aggregate_metrics.call_count == n_global_iter
        assert mock_mlflow_client_log_metrics.call_count == n_global_iter
        assert mock_requires_set_parameters.call_count == n_global_iter
        assert mock_stages_aggregate_parameters.call_count == n_global_iter


def test_get_serverapp(task, driver, node_ids, parameters):
    context = MagicMock()
    filtered_node_ids = [2, 3]

    with (
        patch("fl_server.app.requires.filter_clients") as mock_requires_filter_clients,
        patch("fl_server.app._mlflow_config") as mock_mlflow_config,
        patch("fl_server.app._model_and_data") as mock_model_and_data,
        patch(
            "fl_server.app.requires.get_parameters_from_one_node"
        ) as mock_requires_get_parameters_from_one_node,
        patch("fl_server.app.requires.set_parameters") as mock_requires_set_parameters,
        patch("fl_server.app._training_loop") as mock_training_loop,
        patch("fl_server.app.requires.upload_model") as mock_requires_upload_model,
        patch(
            "fl_server.app.mlflow_client.clean_current_config"
        ) as mock_mlflow_client_clean_current_config,
        patch("fl_server.app.requires.clean_config") as mock_requires_clean_config,
        patch("fl_server.app.rabbitmq_client.setup_rabbitmq") as mock_setup_rabbitmq,
    ):
        mock_requires_filter_clients.return_value = filtered_node_ids
        mock_requires_get_parameters_from_one_node.return_value = parameters
        mock_setup_rabbitmq.return_value = MagicMock()

        app = get_serverapp(task)
        app._main(driver, context)

        mock_requires_filter_clients.assert_called_once_with(
            driver, node_ids, task.use_case
        )
        mock_mlflow_config.assert_called_once_with(task, driver, filtered_node_ids)
        mock_model_and_data.assert_called_once_with(task, driver, filtered_node_ids)
        mock_requires_get_parameters_from_one_node.assert_called_once_with(
            driver, filtered_node_ids
        )
        mock_requires_set_parameters.assert_called_once_with(
            driver, filtered_node_ids, parameters
        )
        mock_training_loop.assert_called_once_with(
            driver, filtered_node_ids, parameters, task.num_global_iterations
        )
        mock_requires_upload_model.assert_called_once_with(
            driver, filtered_node_ids, parameters
        )
        mock_mlflow_client_clean_current_config.assert_called_once()
        mock_requires_clean_config.assert_called_once_with(driver, filtered_node_ids)
