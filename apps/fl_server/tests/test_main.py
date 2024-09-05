import importlib
from unittest.mock import MagicMock, patch

import pytest
from flwr.server import ServerApp

from fl_server.main import configure, event_handler, get_serverapp
from fl_server import config

from schemas.task import TaskRead


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
def mock_rabbitmq_client():
    with patch("fl_server.main.rabbitmq_client") as mock_rabbitmq_client:
        yield mock_rabbitmq_client


@pytest.fixture
def mock_run_server_app():
    with patch("fl_server.main.run_server_app") as mock_run_server_app:
        yield mock_run_server_app


@pytest.fixture
def monkeypatch_env(monkeypatch):
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


def test_main(task, mock_rabbitmq_client, mock_run_server_app):
    mock_listen = mock_rabbitmq_client.listen = MagicMock(return_value=task)
    mock_listen.side_effect = get_serverapp

    event_handler(MagicMock(), MagicMock(), MagicMock(), task.model_dump_json())
    mock_run_server_app.assert_called_once()


def test_configure(monkeypatch_env):
    with (
        patch("fl_server.main.mlflow_client.setup_mlflow") as mock_setup_mlflow,
        patch("fl_server.main.rabbitmq_client.configure") as mock_rabbitmq_configure,
    ):
        configure()

        mock_setup_mlflow.assert_called_once_with(
            monkeypatch_env["MLFLOW_URL"], is_central_node=True
        )

        mock_rabbitmq_configure.assert_called_once_with(
            user=monkeypatch_env["RABBIT_USERNAME"],
            password=monkeypatch_env["RABBIT_PASSWORD"],
            host=monkeypatch_env["RABBIT_HOST"],
            port=int(monkeypatch_env["RABBIT_PORT"]),
        )


def test_event_handler(task):
    with (
        patch("fl_server.main.Task.model_validate_json") as mock_model_validate_json,
        patch("fl_server.main.get_serverapp") as mock_get_serverapp,
        patch("fl_server.main.run_server_app") as mock_run_server_app,
    ):
        mock_model_validate_json.return_value = task
        mock_get_serverapp.return_value = MagicMock(ServerApp)

        event_handler(MagicMock(), MagicMock(), MagicMock(), task.model_dump_json())

        mock_model_validate_json.assert_called_once_with(task.model_dump_json())
        mock_get_serverapp.assert_called_once_with(task)
        mock_run_server_app.assert_called_once_with(mock_get_serverapp.return_value, "")
