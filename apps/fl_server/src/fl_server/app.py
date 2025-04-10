import asyncio
from typing import cast
import structlog
from flwr.common import Context, ParametersRecord
from flwr.server import Driver, ServerApp

from fl_server import requires, stages, config
from fl_server.utils import mlflow_utils
from interfaces import mlflow_client
from schemas.task import TaskRead
from interfaces.rabbitmq_client import AsyncRabbitMQClient

logger = structlog.get_logger()


async def configure_mlflow(task: TaskRead, driver: Driver, node_ids: list[int]) -> None:
    """Configures MLflow experiment runs."""
    if mlflow_client.mlflow.active_run():
        mlflow_client.mlflow.end_run("FAILED")

    experiment_id, parent_run_id, child_run_id = mlflow_utils.create_mlflow_runs(
        task.run_name, task.experiment_name
    )
    mlflow_client.set_current_config(
        experiment_id, parent_run_id, child_run_id, task.model_name, task.model_version
    )

    requires.set_run_config(
        driver,
        node_ids,
        experiment_id,
        parent_run_id,
        task.model_name,
        task.model_version,
    )


async def load_model_and_data(
    task: TaskRead, driver: Driver, node_ids: list[int]
) -> None:
    """Loads model and prepares data for training."""
    stages.load_model(config.global_vars)
    requires.load_data(driver, node_ids, task.use_case)
    requires.load_model(driver, node_ids, task.model_name, task.model_version)
    requires.prepare_data(driver, node_ids)


async def execute_training_loop(
    driver: Driver,
    node_ids: list[int],
    parameters: ParametersRecord,
    n_global_iter: int,
) -> None:
    """Executes the federated training loop with aggregation."""
    for iteration in range(n_global_iter):
        results = requires.train_model(
            driver, node_ids, parameters, current_global_iter=iteration
        )
        aggregated_parameters = stages.aggregate_parameters(
            [r[0] for r in results], config.global_vars
        )
        aggregated_metrics = stages.aggregate_metrics(
            [r[1] for r in results], config.global_vars
        )
        mlflow_client.log_metrics(cast(dict, aggregated_metrics), step=iteration)
        requires.set_parameters(driver, node_ids, aggregated_parameters)


async def server_main(task: TaskRead, driver: Driver, context: Context,
                      rabbitmq: AsyncRabbitMQClient) -> None:
    """Main execution flow for the Federated Learning task."""
    if not task.id or not task.use_case:
        logger.error("Task is missing required fields",
                     task_id=task.id,
                     use_case=task.use_case)
        return
    logger.info("Task started", task_id=task.id)

    node_ids = driver.get_node_ids()
    filtered_node_ids = requires.filter_clients(driver, node_ids,
                                                task.use_case)

    logger.info("Configuring MLflow")
    await configure_mlflow(task, driver, filtered_node_ids)

    logger.info("Loading model and data")
    await load_model_and_data(task, driver, filtered_node_ids)
    port = config.P2P_PORT
    logger.info("Starting P2P servers")
    requires.start_P2P_server(driver, filtered_node_ids, port)

    # call discover_peers_if
    logger.info("Discover peers")
    discovered_peers = requires.discover_peers(driver, filtered_node_ids, port, max_discovery_time=60)

    logger.info("Establishing P2P network")
    requires.connect_peers(driver, filtered_node_ids, discovered_peers)

    logger.info("Getting initial parameters")
    parameters = requires.get_parameters_from_one_node(driver,
                                                       filtered_node_ids)

    logger.info("Broadcasting parameters")
    requires.set_parameters(driver, filtered_node_ids, parameters)

    logger.info("Starting training loop")
    await execute_training_loop(driver, filtered_node_ids, parameters,
                                task.num_global_iterations)

    logger.info("Uploading trained model")
    requires.upload_model(driver, filtered_node_ids, parameters)

    logger.info("Cleaning up MLflow configuration")
    mlflow_client.clean_current_config()
    requires.clean_config(driver, filtered_node_ids)

    logger.info("Task completed successfully", task_id=task.id)


def get_serverapp(task: TaskRead, rabbitmq: AsyncRabbitMQClient) -> ServerApp:
    """Returns a configured ServerApp instance with a synchronous main function."""
    app = ServerApp()

    @app.main()
    def fn(driver: Driver, context: Context) -> None:
        """Runs the FL server synchronously but executes async functions correctly."""
        logger.info("Starting FederatedServerApp...")
        loop = asyncio.get_event_loop()

        if loop.is_running():
            loop.create_task(server_main(task, driver, context, rabbitmq))
        else:
            asyncio.run(server_main(task, driver, context, rabbitmq))

    return app
