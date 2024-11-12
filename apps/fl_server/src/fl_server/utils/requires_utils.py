from flwr.common import Message


def check_success_clients(
    messages: list[Message],
) -> None:
    for msg in messages:
        if msg.has_error():
            print(
                f"Client {msg.metadata.src_node_id} raised error {msg.error.code}: {msg.error.reason}"
            )
        else:
            success = msg.content.configs_records["config"]["success"]
            if not isinstance(success, bool):
                raise TypeError(f"Success field must be a boolean, got {type(success)}")
            if success:
                print(
                    f"Client {msg.metadata.src_node_id} {str(msg.content.configs_records['config']['message'])}"
                )
