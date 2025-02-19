import asyncio
from typing import AsyncGenerator
import uuid
import pytest
import pytest_asyncio
from testcontainers.rabbitmq import RabbitMqContainer
from interfaces.rabbitmq_client import setup_rabbitmq, AsyncRabbitMQClient


@pytest.fixture(scope="session")
def rabbitmq_container():
    """
    This fixture starts a RabbitMQ container for the duration of the test session.
    The container is automatically torn down after the tests complete.
    """
    with RabbitMqContainer("rabbitmq:3-management") as container:
        # Optionally, you can add waiting logic here to ensure RabbitMQ is fully ready.
        yield container


@pytest_asyncio.fixture
async def rabbitmq_client(
    rabbitmq_container,
) -> AsyncGenerator[AsyncRabbitMQClient, None]:
    """
    This fixture initializes the RabbitMQ client using the container's connection details.
    """
    host = rabbitmq_container.get_container_host_ip()
    port = int(rabbitmq_container.get_exposed_port(5672))
    # Testcontainers uses default guest/guest credentials.
    topic = f"test_topic_{uuid.uuid4()}"
    client = setup_rabbitmq(
        user="guest",
        password="guest",
        host=host,
        port=port,
        heartbeat=60,
        topic=topic,
        declare=True,
        exchange="fl-test",
    )
    await client._connect()
    yield client
    await client._close_connection()


@pytest.mark.asyncio
async def test_publish_and_consume(rabbitmq_client: AsyncRabbitMQClient):
    """
    This test verifies that a published message is eventually consumed.
    """
    messages = []

    async def callback(message: str):
        messages.append(message)

    # Start the consumer in a background task.
    consumer_task = asyncio.create_task(rabbitmq_client.listen_forever(callback))

    test_message = "hello world from container"
    await rabbitmq_client.publish_message(test_message)

    # Wait a short time for the message to be processed.
    await asyncio.sleep(1)

    assert test_message in messages

    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_reconnect_publish(rabbitmq_client: AsyncRabbitMQClient):
    """
    This test simulates a connection loss and then verifies that publishing still works.
    """
    # Simulate connection loss.
    await rabbitmq_client._close_connection()

    test_message = "reconnect test from container"
    # The publish_message method should automatically reconnect and send the message.
    await rabbitmq_client.publish_message(test_message)

    messages = []

    async def callback(message: str):
        messages.append(message)

    consumer_task = asyncio.create_task(rabbitmq_client.listen_forever(callback))
    await asyncio.sleep(1)

    assert test_message in messages

    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
