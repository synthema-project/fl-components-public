import asyncio
import argparse
import structlog
from fl_server import config
from fl_server.app import get_serverapp
from fl_server.utils.app_utils import run_server_app
from interfaces import rabbitmq_client, mlflow_client
from schemas.task import TaskRead, TaskStatus, Status

logger = structlog.get_logger()


async def configure() -> dict[str, rabbitmq_client.AsyncRabbitMQClient]:
    """
    Configures MLflow and initializes RabbitMQ for dispatch and status updates.
    Returns a dictionary containing RabbitMQ clients.
    """
    mlflow_client.setup_mlflow(config.MLFLOW_URL, is_central_node=True)

    rabbitmq_clients = {
        "rabbitmq_dispatch": rabbitmq_client.setup_rabbitmq(
            user=config.RABBIT_USERNAME,
            password=config.RABBIT_PASSWORD,
            host=config.RABBIT_HOST,
            port=int(config.RABBIT_PORT),
            heartbeat=30,
            topic="dispatch",
            declare=True,
        ),
        "rabbitmq_status": rabbitmq_client.setup_rabbitmq(
            user=config.RABBIT_USERNAME,
            password=config.RABBIT_PASSWORD,
            host=config.RABBIT_HOST,
            port=int(config.RABBIT_PORT),
            heartbeat=30,
            topic="status",
            declare=True,
        ),
    }

    await rabbitmq_clients["rabbitmq_dispatch"]._connect()
    await rabbitmq_clients["rabbitmq_status"]._connect()

    return rabbitmq_clients


async def _parse_message(message_body: str) -> TaskRead | None:
    """Parses JSON message body into a `TaskRead` object."""
    try:
        task = TaskRead.model_validate_json(message_body)
        return task
    except Exception as e:
        logger.error("Invalid task message received", error=str(e))
        return None


async def _process_task(
    task: TaskRead,
    superlink_url: str,
    rabbitmq_dispatch: rabbitmq_client.AsyncRabbitMQClient,
    rabbitmq_status: rabbitmq_client.AsyncRabbitMQClient,
) -> None:
    """Handles the execution of a federated learning task."""
    try:
        logger.info("Processing task", task_id=task.id)
        app = get_serverapp(task, rabbitmq_dispatch)

        # Publish "running" status
        task_status = TaskStatus(id=task.id, status=Status.RUNNING)
        await rabbitmq_status.publish_message(task_status.model_dump_json())

        # Run the FL process
        await run_server_app(app, superlink_url)

        # Publish "success" status
        task_status = TaskStatus(id=task.id, status=Status.SUCCESS)
        await rabbitmq_status.publish_message(task_status.model_dump_json())

        logger.info("Task completed successfully", task_id=task.id)

    except Exception as e:
        logger.error("Error processing task", task_id=task.id, error=str(e))
        # Optionally publish a failure status
        task_status = TaskStatus(id=task.id, status=Status.FAILED)
        await rabbitmq_status.publish_message(task_status.model_dump_json())


async def event_handler(
    message_body: str,
    superlink_url: str,
    rabbitmq_dispatch: rabbitmq_client.AsyncRabbitMQClient,
    rabbitmq_status: rabbitmq_client.AsyncRabbitMQClient,
) -> None:
    """Handles an incoming RabbitMQ message by parsing and processing the task."""
    task = await _parse_message(message_body)
    if task is None:
        logger.warning("Dropping invalid task message")
        return

    await _process_task(task, superlink_url, rabbitmq_dispatch, rabbitmq_status)


async def main(superlink_url: str) -> None:
    """
    Main entry point for the FL server.
    Subscribes to the 'dispatch' queue and processes tasks indefinitely.
    """
    rabbitmq_clients = await configure()
    dispatch_client = rabbitmq_clients["rabbitmq_dispatch"]
    status_client = rabbitmq_clients["rabbitmq_status"]

    if not dispatch_client._connection or not status_client._connection:
        raise RuntimeError("RabbitMQ connections are not established.")

    logger.info("Starting FL server dispatch consumer...", superlink_url=superlink_url)

    async def dispatch_callback(body: str) -> None:
        """Callback for each 'dispatch' message."""
        await event_handler(body, superlink_url, dispatch_client, status_client)

    # Listen forever on the 'dispatch' queue
    while True:
        try:
            await dispatch_client.listen_forever(dispatch_callback)
        except asyncio.CancelledError:
            logger.info("FL server main loop cancelled.")
            return
        except Exception as e:
            logger.error(f"Dispatch consumer error: {e}")
            await asyncio.sleep(5)  # Retry after a delay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the FL server")
    parser.add_argument(
        "--superlink", type=str, default="0.0.0.0:9091", help="Superlink URL"
    )
    args = parser.parse_args()

    logger.info("Starting FL server", superlink_url=args.superlink)

    try:
        asyncio.run(main(args.superlink))
    except KeyboardInterrupt:
        logger.info("FL server shutting down gracefully")
