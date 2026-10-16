import asyncio
import structlog
from flwr.common import Context, ParametersRecord
from flwr.server import Driver, ServerApp

from fl_server import requires, stages, config
from interfaces.privacy import scalar_metrics
from fl_server.utils import mlflow_utils
from fl_server.interfaces import mlflow_client
from fl_server.schemas.task import TaskRead
from fl_server.interfaces.rabbitmq_client import AsyncRabbitMQClient

logger = structlog.get_logger()


async def configure_mlflow(task: TaskRead, driver: Driver, node_ids: list[int]) -> None:
    """Configures MLflow experiment runs."""
    mlflow_client.clean_current_config("FAILED")

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
    task: TaskRead, driver: Driver, node_ids: list[int],
    node_names: dict[int, str] | None = None,
) -> None:
    """Loads model and prepares data for training."""
    stages.load_model(config.global_vars)
    requires.load_data(
        driver,
        node_ids,
        task.use_case,
        config.global_vars.get("metadata_documents"),
        node_names=node_names,
        **({"dataset_snapshot": getattr(task, "validation_snapshot", None)} if getattr(task, "validation_snapshot", None) else {}),
    )
    requires.execute_step_zero(driver, node_ids, config.global_vars)
    applied = requires.exchange_custom_metadata(driver, node_ids, config.global_vars)
    logger.info("Custom metadata exchange completed", applied=applied)
    requires.prepare_data(driver, node_ids)

async def set_SMPC_enabled(
    task: TaskRead, driver: Driver, node_ids: list[int]
) -> None:
    """Sets SMPC enabled flag for the task."""
    stages.set_SMPC_enabled(task.SMPC_enabled, config.global_vars)
    requires.set_SMPC_enabled(driver, node_ids, task.SMPC_enabled)

async def set_dp_config(
    task: TaskRead, driver: Driver, node_ids: list[int]
) -> None:
    stages.configure_dp(config.global_vars,task.dp_mode,task.noise_multiplier,task.clipping_norm, len(node_ids))
    requires.set_dp_config(driver, node_ids, task.dp_mode,task.noise_multiplier,task.clipping_norm)


async def execute_training_loop(
    driver: Driver,
    node_ids: list[int],
    parameters: ParametersRecord,
    n_global_iter: int,
) -> ParametersRecord:
    """Return the final aggregate for subsequent rounds and model upload."""
    if type(n_global_iter) is not int or n_global_iter < 1:
        raise ValueError("num_global_iterations must be a positive integer")
    for iteration in range(n_global_iter):
        results = requires.train_model(
            driver, node_ids, parameters, current_global_iter=iteration, max_round=n_global_iter
        )
        aggregated_parameters = stages.aggregate_parameters(
            [r[0] for r in results], config.global_vars
        )
        aggregated_metrics = stages.aggregate_metrics(
            [r[1] for r in results], config.global_vars
        )
        mlflow_client.log_metrics({**scalar_metrics(aggregated_metrics), "fl/current_round": iteration + 1, "fl/max_round": n_global_iter}, step=iteration)
        requires.set_parameters(driver, node_ids, aggregated_parameters)
        evaluation = requires.evaluate_model(
            driver, node_ids, aggregated_parameters,
            current_global_iter=iteration, max_round=n_global_iter,
        )
        if evaluation:
            evaluation_metrics = stages.aggregate_metrics(evaluation, config.global_vars)
            mlflow_client.log_metrics(
                {f"eval/{key}": value for key, value in scalar_metrics(evaluation_metrics).items()},
                step=iteration,
            )
        parameters = aggregated_parameters
    return parameters


async def server_main(task: TaskRead, driver: Driver, context: Context,
                      rabbitmq: AsyncRabbitMQClient) -> None:
    """Main execution flow for the Federated Learning task."""
    if not task.id or not task.use_case:
        logger.error("Task is missing required fields",
                     task_id=task.id,
                     use_case=task.use_case)
        return
    logger.info("Task started", task_id=task.id, num_synthetic_samples=task.num_synthetic_samples) 

    import os
    if os.getenv('VALIDATION_PROFILE') in {'aml-full', 'scd-full'} and not getattr(task, "validation_snapshot", None):
        raise ValueError('Full validation requires a task configuration snapshot; submit through the updated API')
    node_ids = driver.get_node_ids()
    node_names: dict[int, str] = {}
    filtered_node_ids = requires.filter_clients(
        driver, node_ids, task.use_case, task.selected_nodes, node_names
    )

    if not filtered_node_ids:
        raise ValueError('Training requires at least one participating client')
    if task.SMPC_enabled and len(filtered_node_ids) < 2:
        raise ValueError('SMPC training requires at least two participating clients')

    try:
        logger.info("Configuring MLflow")
        await configure_mlflow(task, driver, filtered_node_ids)

        logger.info("Loading model and data")
        await load_model_and_data(task, driver, filtered_node_ids, node_names)

        # Reset both server and clients on every task, including after a legacy run.
        await set_SMPC_enabled(task, driver, filtered_node_ids)
        if task.SMPC_enabled:
            port = config.P2P_PORT
            logger.info("Starting P2P servers")
            requires.start_P2P_server(driver, filtered_node_ids, port)

            logger.info("Discover peers")
            discovered_peers = requires.discover_peers(driver, filtered_node_ids, port, max_discovery_time=60)

            logger.info("Establishing P2P network")
            requires.connect_peers(driver, filtered_node_ids, discovered_peers)

        # Configure or clear the existing optional platform DP on every task.
        await set_dp_config(task, driver, filtered_node_ids)

        logger.info("Getting initial parameters")
        parameters = requires.get_parameters_from_one_node(driver,
                                                           filtered_node_ids)

        logger.info("Broadcasting parameters")
        requires.set_parameters(driver, filtered_node_ids, parameters)

        logger.info("Starting training loop")
        parameters = await execute_training_loop(driver, filtered_node_ids, parameters,
                                    task.num_global_iterations)

        privacy = requires.collect_privacy_budgets(driver, filtered_node_ids)
        config.global_vars['privacy_result'] = privacy
        privacy_metrics = {
            'privacy/reported_nodes': privacy['reported_nodes'],
            'privacy/total_nodes': privacy['total_nodes'],
            'privacy/budget_complete': int(privacy['complete']),
        }
        if 'model_reported_budget' in privacy:
            privacy_metrics.update({f'privacy/model_reported_{key}': value
                                    for key, value in privacy['model_reported_budget'].items()})
        mlflow_client.log_metrics(privacy_metrics, step=task.num_global_iterations - 1)
        logger.info('Model privacy budget collection completed', **privacy)

        logger.info("Uploading trained model")
        requires.upload_model(driver, filtered_node_ids, parameters)

        if task.num_synthetic_samples > 0:
            logger.info("Generating and publishing synthetic data")
            requires.generate_synthetic_data(
                driver,
                filtered_node_ids,
                task.num_synthetic_samples,
                task.use_case,
                task.model_name,
                node_names=node_names,
                **({"dataset_snapshot": getattr(task, "validation_snapshot", None), "task_id": task.id} if getattr(task, "validation_snapshot", None) else {}),
            )

        requires.clean_config(driver, filtered_node_ids)
        mlflow_client.clean_current_config()
    except Exception:
        # Cleanup runs in the same worker as MLflow's thread-local active run.
        # Preserve the original failure if tracking or a client is unavailable.
        try:
            requires.clean_config(driver, filtered_node_ids, status="FAILED")
        except Exception:
            logger.exception("Client MLflow cleanup failed", task_id=task.id)
        try:
            mlflow_client.clean_current_config("FAILED")
        except Exception:
            logger.exception("Server MLflow cleanup failed", task_id=task.id)
        raise

    logger.info("Task completed successfully", task_id=task.id)


def get_serverapp(task: TaskRead, rabbitmq: AsyncRabbitMQClient) -> ServerApp:
    """Returns a configured ServerApp instance with a synchronous main function."""
    app = ServerApp()

    @app.main()
    def fn(driver: Driver, context: Context) -> None:
        """Runs the FL server synchronously but executes async functions correctly."""
        logger.info("Starting FederatedServerApp...")
        # The dispatcher invokes this callback in an asyncio.to_thread worker.
        # Wait for completion so failures propagate to the task dispatcher.
        asyncio.run(server_main(task, driver, context, rabbitmq))

    return app
