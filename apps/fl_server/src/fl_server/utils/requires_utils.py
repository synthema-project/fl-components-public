from flwr.common import Message


def check_success_clients(messages: list[Message]) -> None:
    """Fail at the original stage instead of treating error replies as data."""
    for msg in messages:
        node = msg.metadata.src_node_id
        if msg.has_error():
            raise RuntimeError(f"Client {node} raised error {msg.error.code}: {msg.error.reason}")
        if not msg.has_content():
            raise RuntimeError(f"Client {node} returned no content")
        config = msg.content.configs_records.get("config", None)
        if config is None:
            raise RuntimeError(f"Client {node} returned no status record")
        success = config.get("success", None)
        if not isinstance(success, bool):
            raise TypeError(f"Client {node}: success must be boolean, got {type(success)}")
        if not success:
            raise RuntimeError(f"Client {node} failed: {config.get('message', 'unspecified failure')}")
        print(f"Client {node} {config.get('message', 'completed successfully')}")
