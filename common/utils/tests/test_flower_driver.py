import pytest
from unittest.mock import Mock, patch

from flwr.common import DEFAULT_TTL, Message, RecordSet, Metadata
from flwr.server.driver import GrpcDriver as Driver

from utils.src.flower_utils.driver import (
    create_messages,
    send_messages,
    wait_messages,
)


@pytest.fixture(scope="session")
def driver():
    return Mock(spec=Driver)


@pytest.fixture(scope="session")
def content():
    return RecordSet()


@pytest.fixture(scope="session")
def message_type():
    return "test_message"


@pytest.fixture(scope="session")
def dst_node_id():
    return [1]


@pytest.fixture(scope="session")
def group_id():
    return "test_group"


@pytest.fixture(scope="session")
def ttl():
    return DEFAULT_TTL


@pytest.fixture(scope="session")
def messages(content):
    return [Message(metadata=Mock(spec=Metadata), content=content) for _ in range(3)]


@pytest.fixture(scope="session")
def message_ids():
    return ["1", "2", "3"]


def test_create_messages(driver, content, message_type, dst_node_id, group_id, ttl):
    with patch.object(driver, "create_message") as mock_create_message:
        _ = create_messages(driver, content, message_type, dst_node_id, group_id, ttl)
    mock_create_message.assert_called_with(
        content=content,
        message_type=message_type,
        dst_node_id=dst_node_id[0],
        group_id=group_id,
        ttl=ttl,
    )


def test_send_messages(driver, messages):
    with patch.object(driver, "push_messages") as mock_push_messages:
        _ = send_messages(driver, messages)
    mock_push_messages.assert_called_with(messages)


def test_wait_messages(driver, message_ids):
    with patch.object(
        driver, "pull_messages", return_value=["1", "2", "3"]
    ) as mock_pull_messages:
        _ = wait_messages(driver, message_ids)
    mock_pull_messages.assert_called_with(message_ids=message_ids)
