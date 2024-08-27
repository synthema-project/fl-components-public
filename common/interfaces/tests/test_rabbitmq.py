import pytest
from unittest.mock import patch, Mock

import pika

from interfaces import rabbitmq_client


@pytest.fixture(autouse=True)
def reset_rabbitmq_client():
    # Reset global variables before each test
    rabbitmq_client._connection = None
    rabbitmq_client._channel = None
    rabbitmq_client._queue_name = None
    rabbitmq_client._is_configured.value = False


def test_configure_rabbitmq_client():
    with patch(
        "interfaces.rabbitmq_client.pika.BlockingConnection"
    ) as mock_blocking_connection:
        mock_connection = Mock()
        mock_channel = Mock()
        mock_blocking_connection.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel

        rabbitmq_client.configure(
            user="guest", password="guest", host="localhost", queue_name="fml-engine"
        )

        mock_blocking_connection.assert_called_once_with(
            pika.ConnectionParameters(
                host="localhost", credentials=pika.PlainCredentials("guest", "guest")
            )
        )
        mock_connection.channel.assert_called_once()
        mock_channel.queue_declare.assert_called_once_with(
            queue="fml-engine", durable=True
        )

        assert rabbitmq_client._is_configured
        assert rabbitmq_client._queue_name == "fml-engine"


def test_ensure_configured():
    with pytest.raises(RuntimeError):
        rabbitmq_client.publish_message("test")

    rabbitmq_client._is_configured.value = True
    rabbitmq_client._channel = Mock()

    try:
        rabbitmq_client.publish_message("test")
    except RuntimeError:
        pytest.fail("publish_message raised RuntimeError unexpectedly!")


@patch("interfaces.rabbitmq_client._channel")
def test_publish_message(mock_channel):
    rabbitmq_client._is_configured.value = True
    rabbitmq_client._queue_name = "fml-engine"

    rabbitmq_client.publish_message("Hello, RabbitMQ!")

    mock_channel.basic_publish.assert_called_once_with(
        exchange="", routing_key="fml-engine", body="Hello, RabbitMQ!"
    )


@patch("interfaces.rabbitmq_client._channel")
def test_listen(mock_channel):
    rabbitmq_client._is_configured.value = True
    rabbitmq_client._queue_name = "fml-engine"

    callback = Mock()
    rabbitmq_client.listen(callback)

    mock_channel.basic_consume.assert_called_once_with(
        queue="fml-engine", on_message_callback=callback, auto_ack=True
    )
    mock_channel.start_consuming.assert_called_once()


@patch("interfaces.rabbitmq_client._connection")
@patch("interfaces.rabbitmq_client._channel")
def test_close_connection(mock_channel, mock_connection):
    rabbitmq_client._is_configured.value = True

    rabbitmq_client._close_connection()

    mock_channel.close.assert_called_once()
    mock_connection.close.assert_called_once()
