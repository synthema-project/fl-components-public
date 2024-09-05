from pika import spec
from pika.channel import Channel

from fl_server import config
from fl_server.app import get_serverapp
from fl_server.utils.app_utils import run_server_app

from interfaces import rabbitmq_client, mlflow_client
from schemas.task import Task

SUPERLINK_URL = ""


def configure() -> None:
    mlflow_client.setup_mlflow(config.MLFLOW_URL, is_central_node=True)
    rabbitmq_client.configure(
        user=config.RABBIT_USERNAME,
        password=config.RABBIT_PASSWORD,
        host=config.RABBIT_HOST,
        port=int(config.RABBIT_PORT),
    )


def event_handler(
    channel: Channel,
    method_frame: spec.Basic.Deliver,
    header_frame: spec.Basic.Deliver,
    body: bytes,
) -> None:
    task = Task.model_validate_json(body)
    app = get_serverapp(task)
    run_server_app(app, SUPERLINK_URL)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Start the FL server")
    parser.add_argument(
        "--superlink", type=str, default="0.0.0.0:9091", help="Superlink URL"
    )

    args = parser.parse_args()
    SUPERLINK_URL = args.superlink

    configure()
    while True:
        rabbitmq_client.listen(event_handler)
