# import json
# import pytest
# from unittest.mock import AsyncMock, MagicMock, patch
# import pytest_asyncio

# from restapi.services.status_updates import StatusUpdates
# from schemas.task import Task, Status


# @pytest_asyncio.fixture
# def mock_rabbitmq():
#     """Creates a mock RabbitMQ client."""
#     mock = AsyncMock()

#     async def listen_mock():
#         """Simulates async message listening."""
#         yield json.dumps({"task_id": 1, "status": Status.SUCCESS.value})
#         yield json.dumps({"task_id": "invalid", "status": "wrong_status"})

#     mock.listen = listen_mock
#     mock.publish_message = AsyncMock()
#     return mock


# @pytest_asyncio.fixture
# async def mock_session():
#     """Creates a mock async database session."""
#     session = AsyncMock()
#     session.get.return_value = MagicMock(spec=Task)  # Mock Task object
#     session.commit = AsyncMock()
#     session.rollback = AsyncMock()
#     return session


# @pytest_asyncio.fixture
# def mock_config(mock_rabbitmq):
#     """Mocks the global config to inject the mocked RabbitMQ client."""
#     with patch("restapi.config.obj", {"rabbitmq_status": mock_rabbitmq}):
#         yield


# @pytest.mark.asyncio
# async def test_status_updates_success(mock_config, mock_session):
#     """Tests that a valid status update is processed correctly."""

#     async def get_session_mock():
#         """Mock get_session() to properly yield the session."""
#         yield mock_session

#     with patch("restapi.services.status_updates.get_session", get_session_mock):
#         status_updates = StatusUpdates()
#         await status_updates._update_task_status(
#             mock_session,
#             mock_session.get.return_value,
#             {"task_id": 1, "status": "success"},
#         )

#         # Assertions
#         mock_session.get.assert_called_once_with(Task, 1)
#         task = mock_session.get.return_value
#         task.change_status.assert_called_once_with("success")
#         mock_session.commit.assert_called_once()


# @pytest.mark.asyncio
# async def test_status_updates_invalid_message(mock_config, mock_session):
#     """Tests that an invalid message is logged and ignored."""

#     with patch(
#         "restapi.services.status_updates.get_session", return_value=mock_session
#     ):
#         status_updates = StatusUpdates()

#         with patch("restapi.services.status_updates.logger.error") as mock_logger:
#             await status_updates._handle_status_message(
#                 json.dumps({"task_id": "invalid", "status": "wrong_status"})
#             )

#             # Ensure an error is logged
#             mock_logger.assert_called_once()

#         # Ensure DB was not modified
#         mock_session.get.assert_not_called()


# @pytest.mark.asyncio
# async def test_status_updates_requeues_failed_task(
#     mock_config, mock_rabbitmq, mock_session
# ):
#     """Tests that a failed task update is requeued."""
#     with patch(
#         "restapi.services.status_updates.get_session", return_value=mock_session
#     ):
#         status_updates = StatusUpdates()

#         # Simulate a DB error
#         mock_session.commit.side_effect = Exception("DB Failure")

#         await status_updates._update_task_status(
#             mock_session,
#             mock_session.get.return_value,
#             {"task_id": 1, "status": "failed"},
#         )

#         # Ensure rollback and requeue are called
#         mock_session.rollback.assert_called_once()
#         mock_rabbitmq.publish_message.assert_called_once()
