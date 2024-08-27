from fl_server import config
from fl_server.app import get_serverapp

from utils.src.flower_utils.server_app import run_server_app
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


def event_handler(channel, method_frame, header_frame, body):
    task = Task.model_validate_json(body)
    app = get_serverapp(task)
    run_server_app(app)


if __name__ == "__main__":
    configure()
    while True:
        rabbitmq_client.listen(event_handler)
