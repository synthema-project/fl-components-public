import importlib
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import pytest_asyncio
import aio_pika

from fl_server.main import event_handler, configure
from fl_server import config

from schemas.task import TaskRead


@pytest_asyncio.fixture
def task():
    """Provides a sample task for testing."""
    return TaskRead(
        user_id="user_id",
        use_case="use_case",
        model_name="model_name",
        model_version=1,
        num_global_iterations=2,
        id=1,
        status="pending",
        created_at="2021-08-01T00:00:00",  # type: ignore
        run_name="run_name",
        experiment_name="experiment_name",
    )


@pytest_asyncio.fixture
async def mock_rabbitmq_client():
    """Mocks the AsyncRabbitMQClient."""
    mock_client = AsyncMock()
    mock_client._connect = AsyncMock()
    yield mock_client


@pytest_asyncio.fixture
async def mock_run_server_app():
    """Mocks the `run_server_app` function."""
    with patch(
        "fl_server.main.run_server_app", new_callable=AsyncMock
    ) as mock_run_server_app:
        yield mock_run_server_app


@pytest_asyncio.fixture
def monkeypatch_env(monkeypatch):
    """Mocks environment variables for the test run."""
    envvars = {
        "MLFLOW_URL": "mock_mlflow_url",
        "RABBIT_USERNAME": "mock_rabbit_username",
        "RABBIT_PASSWORD": "mock_rabbit_password",
        "RABBIT_HOST": "mock_rabbit_host",
        "RABBIT_PORT": "10",
    }

    for key, value in envvars.items():
        monkeypatch.setenv(key, value)

    importlib.reload(config)

    yield envvars


@pytest.mark.asyncio
async def test_configure(monkeypatch_env):
    """Tests that configure() correctly sets up MLflow and RabbitMQ."""
    with (
        patch(
            "fl_server.main.mlflow_client.setup_mlflow", new_callable=AsyncMock
        ) as mock_setup_mlflow,
        patch(
            "fl_server.main.rabbitmq_client.setup_rabbitmq", return_value=AsyncMock
        ) as mock_rabbitmq_setup,
    ):
        mock_rabbitmq_client = AsyncMock()
        mock_rabbitmq_setup.return_value = mock_rabbitmq_client
        await configure()

        mock_setup_mlflow.assert_called_once_with(
            monkeypatch_env["MLFLOW_URL"], is_central_node=True
        )

        mock_rabbitmq_setup.assert_has_calls(
            [
                call(
                    user=monkeypatch_env["RABBIT_USERNAME"],
                    password=monkeypatch_env["RABBIT_PASSWORD"],
                    host=monkeypatch_env["RABBIT_HOST"],
                    port=int(monkeypatch_env["RABBIT_PORT"]),
                    heartbeat=30,
                    topic="status",
                    declare=True,
                ),
                call(
                    user=monkeypatch_env["RABBIT_USERNAME"],
                    password=monkeypatch_env["RABBIT_PASSWORD"],
                    host=monkeypatch_env["RABBIT_HOST"],
                    port=int(monkeypatch_env["RABBIT_PORT"]),
                    heartbeat=30,
                    topic="dispatch",
                    declare=True,
                ),
            ],
            any_order=True,
        )


@pytest.mark.asyncio
async def test_event_handler(task, mock_rabbitmq_client):
    """Tests that `event_handler` correctly processes RabbitMQ messages and starts FL server."""
    with (
        patch(
            "fl_server.main._parse_message", new_callable=AsyncMock
        ) as mock_parse_message,
        patch(
            "fl_server.main._process_task", new_callable=AsyncMock
        ) as mock_process_task,
    ):
        mock_parse_message.return_value = task
        mock_process_task.return_value = None

        mock_rabbitmq_client_dispatch = mock_rabbitmq_client
        mock_rabbitmq_client_status = mock_rabbitmq_client

        # Simulate an incoming RabbitMQ message
        message = MagicMock(aio_pika.IncomingMessage)
        message.body = task.model_dump_json().encode()

        await event_handler(
            message,
            "mock_superlink_url",
            mock_rabbitmq_client_dispatch,
            mock_rabbitmq_client_status,
        )

        # Verify the parsing and processing calls
        mock_parse_message.assert_called_once_with(message)
        mock_process_task.assert_called_once_with(
            task,
            "mock_superlink_url",
            mock_rabbitmq_client_dispatch,
            mock_rabbitmq_client_status,
        )
