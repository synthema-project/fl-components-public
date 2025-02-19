import structlog
from status_listener.db import get_session
from schemas.task import TaskRead, TaskStatus
from status_listener import config
from interfaces.rabbitmq_client import setup_rabbitmq

logger = structlog.get_logger()

rabbitmq_client = setup_rabbitmq(
    user=config.RABBIT_USERNAME,
    password=config.RABBIT_PASSWORD,
    host=config.RABBIT_HOST,
    port=int(config.RABBIT_PORT),
    heartbeat=60,
    topic=config.RABBIT_QUEUE,
    declare=True,
)


async def process_status_update(message: str) -> None:
    logger.info("Processing message", extra={"message": message})
    try:
        parsed_status = TaskStatus.model_validate_json(message)
        task_id = parsed_status.id
        new_status = parsed_status.status

        async for session in get_session():
            task = await session.get(TaskRead, int(task_id))
            if task:
                task.change_status(new_status)
                await session.commit()
                logger.info(f"Task {task_id} updated to {new_status}")
            else:
                logger.warning(f"Task {task_id} not found in DB")
    except Exception as e:
        logger.error("Failed to process status update", extra={"error": str(e)})


async def consume_status_updates() -> None:
    """Continuously listens for RabbitMQ status updates."""
    logger.info("Starting to consume status updates infinitely...")
    await rabbitmq_client.listen_forever(process_status_update)
