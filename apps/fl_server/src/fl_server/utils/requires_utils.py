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
            config = msg.content.configs_records["config"]
            success = config["success"]
            if not isinstance(success, bool):
                raise TypeError(f"Success field must be a boolean, got {type(success)}")
            if success:
                # Not every success reply carries a human-readable "message"
                # (e.g. generate_synthetic_data returns "synthetic_csv" instead),
                # so fall back gracefully rather than KeyError.
                detail = config["message"] if "message" in config else "completed successfully"
                print(f"Client {msg.metadata.src_node_id} {detail}")
