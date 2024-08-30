from pika import spec
from pika.channel import Channel

from fl_server import config
from fl_server.app import get_serverapp
from fl_server.utils.app_utils import run_server_app

from interfaces import rabbitmq_client, mlflow_client
from schemas.task import Task


def configure() -> None:
    mlflow_client.setup_mlflow(config.MLFLOW_URL)
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
    run_server_app(app)


if __name__ == "__main__":  # pragma: no cover
    configure()
    while True:
        rabbitmq_client.listen(event_handler)
