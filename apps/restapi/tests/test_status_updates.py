from threading import Event
import time
import pytest
from unittest.mock import MagicMock, patch
from restapi.status_updates import StatusUpdates
from schemas.task import Task


@pytest.fixture
def mock_rabbitmq():
    mock = MagicMock()
    mock.listen.return_value = "task 1 success"
    return mock


@pytest.fixture
def mock_session():
    with patch("restapi.status_updates.Session", autospec=True) as mock_session:
        yield mock_session


@pytest.fixture
def mock_task():
    mock = MagicMock(spec=Task)
    return mock


def test_status_updates(mock_rabbitmq, mock_session, mock_task):
    # Mock config to return our mock RabbitMQ client
    with patch("restapi.status_updates.config") as mock_config:
        mock_config.obj = {"rabbitmq_status": mock_rabbitmq}

        # Mock the database session and task
        mock_session.return_value.__enter__.return_value.get.return_value = mock_task

        # Create and run the StatusUpdates thread
        event = Event()
        status_updates = StatusUpdates(event)
        status_updates.start()
        time.sleep(1)

        # Assertions
        mock_rabbitmq.listen.assert_called_once_with(None, blocking=False)
        mock_session.return_value.__enter__.return_value.get.assert_called_once_with(
            Task, 1
        )
        mock_task.change_status.assert_called_once_with("success")
        mock_session.return_value.__enter__.return_value.commit.assert_called_once()
        mock_session.return_value.__enter__.return_value.refresh.assert_called_once_with(
            mock_task
        )

        event.set()
        status_updates.join()
