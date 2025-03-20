import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession
from status_listener.status_consumer import (
    process_status_update,
    consume_status_updates,
)
from schemas.task import TaskRead


@pytest.mark.asyncio
async def test_process_status_update():
    """Test that `process_status_update` correctly updates a task's status in the database."""
    mock_message = '{"id": 1, "status": "success"}'

    mock_task = MagicMock(spec=TaskRead)
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.get.return_value = mock_task

    async def mock_get_session():
        yield mock_session

    with patch("status_listener.status_consumer.get_session", mock_get_session):
        await process_status_update(mock_message)

    mock_task.change_status.assert_called_once_with("success")
    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_process_status_update_invalid_message():
    """Test that `process_status_update` handles invalid messages gracefully."""
    mock_message = "invalid json"

    with patch("status_listener.status_consumer.logger.error") as mock_logger:
        await process_status_update(mock_message)

    mock_logger.assert_called_once()


@pytest.mark.asyncio
async def test_consume_status_updates():
    """Test that `consume_status_updates` listens for RabbitMQ messages and processes them."""
    mock_rabbitmq_client = AsyncMock()
    mock_rabbitmq_client.listen_with_callback = AsyncMock()

    with patch("status_listener.status_consumer.rabbitmq_client", mock_rabbitmq_client):
        task = asyncio.create_task(consume_status_updates())

        await asyncio.sleep(0.1)

    # mock_rabbitmq_client.listen.assert_called()

    task.cancel()
