import time

from flwr.common import DEFAULT_TTL, Message, RecordSet
from flwr.server import Driver


def create_messages(
    driver: Driver,
    content: RecordSet,
    message_type: str,
    dst_node_id: list[int],
    group_id: str,
    ttl: int = DEFAULT_TTL,
) -> list[Message]:
    messages = []
    for node_id in dst_node_id:
        message = driver.create_message(
            content=content,
            message_type=message_type,
            dst_node_id=node_id,
            group_id=group_id,
            ttl=ttl,
        )
        messages.append(message)
    return messages


def send_messages(driver: Driver, messages: list[Message]) -> list[str]:
    message_ids = list(driver.push_messages(messages))
    print(f"Pushed {len(message_ids)} messages: {message_ids}")
    return message_ids


def wait_messages(driver: Driver, message_ids: list[str]) -> list[Message]:
    message_ids = [message_id for message_id in message_ids if message_id != ""]
    all_replies: list[Message] = []
    while True:
        replies = list(driver.pull_messages(message_ids=message_ids))
        print(f"Got {len(replies)} results")
        all_replies += replies
        if len(all_replies) == len(message_ids):
            break
        time.sleep(3)
    return all_replies
