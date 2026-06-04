import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flwr.common import ParametersRecord, MetricsRecord
from fl_server.app import (
    configure_mlflow,
    load_model_and_data,
    execute_training_loop,
    get_serverapp,
)
from fl_server import config
from fl_server.schemas.task import TaskRead


@pytest.fixture
def task():
    return TaskRead(
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


@pytest.mark.asyncio
async def test_configure_mlflow(task, driver, node_ids):
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
        await configure_mlflow(task, driver, node_ids)

        mock_create_runs.assert_called_once_with(task.run_name, task.experiment_name)
        mock_set_config.assert_called_once()
        mock_set_run_config.assert_called_once()


@pytest.mark.asyncio
async def test_load_model_and_data(task, driver, node_ids):
    with (
        patch("fl_server.app.stages.load_model") as mock_load_model,
        patch("fl_server.app.requires.load_data") as mock_load_data,
        patch("fl_server.app.requires.load_model") as mock_load_model_driver,
        patch("fl_server.app.requires.prepare_data") as mock_prepare_data,
        patch.dict(config.global_vars, {"aggregator": "agg"}),
    ):
        await load_model_and_data(task, driver, node_ids)

        mock_load_model.assert_called_once()
        mock_load_data.assert_called_once_with(driver, node_ids, task.use_case)
        mock_load_model_driver.assert_called_once_with(
            driver, node_ids, task.model_name, task.model_version
        )
        mock_prepare_data.assert_called_once_with(driver, node_ids)


@pytest.mark.asyncio
async def test_execute_training_loop(driver, node_ids, parameters):
    n_global_iter = 5
    with (
        patch("fl_server.app.requires.train_model") as mock_train,
        patch("fl_server.app.stages.aggregate_parameters") as mock_aggregate_params,
        patch("fl_server.app.stages.aggregate_metrics") as mock_aggregate_metrics,
        patch("fl_server.app.mlflow_client.log_metrics") as mock_log_metrics,
        patch("fl_server.app.requires.set_parameters") as mock_set_params,
        patch("fl_server.app.requires.evaluate_model") as mock_evaluate,
    ):
        mock_train.return_value = [(parameters, {}) for _ in range(n_global_iter)]
        mock_aggregate_params.return_value = parameters
        mock_evaluate.return_value = []
        mock_aggregate_metrics.return_value = MetricsRecord({"accuracy": 0.9})

        await execute_training_loop(driver, node_ids, parameters, n_global_iter)

        assert mock_train.call_count == n_global_iter
        assert mock_aggregate_params.call_count == n_global_iter
        assert mock_evaluate.call_count == n_global_iter
        assert mock_log_metrics.call_count == n_global_iter * 2
        assert mock_set_params.call_count == n_global_iter



@pytest.mark.asyncio
async def test_federated_server_app(task, driver, node_ids, parameters):
    """Tests that the `FederatedServerApp` executes correctly."""
    context = MagicMock()
    filtered_node_ids = [2, 3]

    with (
        patch(
            "fl_server.app.requires.filter_clients", autospec=True
        ) as mock_filter_clients,
        patch("fl_server.app.configure_mlflow", autospec=True) as mock_mlflow_config,
        patch(
            "fl_server.app.load_model_and_data", autospec=True
        ) as mock_model_and_data,
        patch(
            "fl_server.app.requires.get_parameters_from_one_node", autospec=True
        ) as mock_get_parameters,
        patch(
            "fl_server.app.requires.set_parameters", autospec=True
        ) as mock_set_parameters,
        patch(
            "fl_server.app.execute_training_loop", autospec=True
        ) as mock_training_loop,
        patch(
            "fl_server.app.requires.upload_model", autospec=True
        ) as mock_upload_model,
        patch(
            "fl_server.app.mlflow_client.clean_current_config", autospec=True
        ) as mock_clean_mlflow,
        patch(
            "fl_server.app.requires.clean_config", autospec=True
        ) as mock_clean_config,
        patch(
            "fl_server.app.mlflow_client.mlflow.active_run", return_value=False
        ),  # ✅ Mock MLflow correctly
    ):
        mock_filter_clients.return_value = filtered_node_ids
        mock_get_parameters.return_value = parameters

        rabbitmq = AsyncMock()
        rabbitmq.publish_message = AsyncMock()

        app = get_serverapp(task, rabbitmq)

        assert hasattr(app, "_main"), "FederatedServerApp must define `_main`"
        assert callable(app._main), "FederatedServerApp._main should be callable"

        app._main(driver, context)  # Start the task

        # ✅ Retrieve the running task and wait for it to complete
        loop = asyncio.get_event_loop()
        tasks = [t for t in asyncio.all_tasks(loop) if "server_main" in str(t)]

        for running_task in tasks:
            await running_task

        mock_filter_clients.assert_called_once_with(driver, node_ids, task.use_case)
        mock_mlflow_config.assert_called_once_with(task, driver, filtered_node_ids)
        mock_model_and_data.assert_called_once_with(task, driver, filtered_node_ids)
        mock_get_parameters.assert_called_once_with(driver, filtered_node_ids)
        mock_set_parameters.assert_called_once()
        mock_training_loop.assert_called_once_with(
            driver, filtered_node_ids, parameters, task.num_global_iterations
        )
        mock_upload_model.assert_called_once_with(driver, filtered_node_ids, parameters)
        mock_clean_mlflow.assert_called_once()
        mock_clean_config.assert_called_once_with(driver, filtered_node_ids)
