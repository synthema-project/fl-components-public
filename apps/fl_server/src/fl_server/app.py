import asyncio
import json
import random
from typing import cast
import pandas as pd
import structlog
from flwr.common import Context, ParametersRecord
from flwr.server import Driver, ServerApp

from fl_server import requires, stages, config
from fl_server.utils import mlflow_utils
from fl_server.interfaces import mlflow_client
from fl_server.schemas.task import TaskRead
from fl_server.interfaces.rabbitmq_client import AsyncRabbitMQClient
from model_store_interface.core.schemas.dataset_meta import DatasetMeta
from model_store_interface.catalogue_hook import publish_to_catalogue

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
        task.dataset_meta_json,
    )


async def load_model_and_data(
    task: TaskRead, driver: Driver, node_ids: list[int]
) -> None:
    """Loads model and prepares data for training."""
    stages.load_model(config.global_vars)
    requires.load_data(driver, node_ids, task.use_case)
    requires.load_model(driver, node_ids, task.model_name, task.model_version)
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
        parameters = aggregated_parameters

        # Trigger intra-round evaluation (internal validation)
        eval_metrics = requires.evaluate_model(driver, node_ids, aggregated_parameters)
        aggregated_eval_metrics = stages.aggregate_metrics(
            eval_metrics, config.global_vars
        )
        val_metrics_dict = {f"val_{k}": v for k, v in aggregated_eval_metrics.items()}
        mlflow_client.log_metrics(val_metrics_dict, step=iteration)



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

    if task.SMPC_enabled:
        await set_SMPC_enabled(task, driver, filtered_node_ids)
        port = config.P2P_PORT
        logger.info("Starting P2P servers")
        requires.start_P2P_server(driver, filtered_node_ids, port)

        logger.info("Discover peers")
        discovered_peers = requires.discover_peers(driver, filtered_node_ids, port, max_discovery_time=60)

        logger.info("Establishing P2P network")
        requires.connect_peers(driver, filtered_node_ids, discovered_peers)

    if task.dp_mode:
        logger.info(f"Configuring DP {task.dp_mode} with clipping norm {task.clipping_norm} and noise multiplier {task.noise_multiplier}")
        await set_dp_config(task, driver, filtered_node_ids)

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

    if task.dataset_meta_json:
        logger.info("Starting post-training validation pipeline")
        try:
            dataset_meta = DatasetMeta.model_validate(json.loads(task.dataset_meta_json))

            # Open a sibling postprocessing MLflow run
            experiment_id = str(mlflow_client._current_config.experiment_id)
            parent_run_id = str(mlflow_client._current_config.parent_run_id)
            post_run_id = mlflow_client.create_postprocessing_run(experiment_id, parent_run_id)

            # Elect one node for synthetic data generation
            generator_node_id = random.choice(filtered_node_ids)
            logger.info("Generating synthetic data", generator_node=generator_node_id)
            node_csvs = requires.generate_synthetic_data(driver, [generator_node_id], n_samples=1000)
            synthetic_csv = list(node_csvs.values())[0]

            # Broadcast synthetic CSV to all nodes for federated validation
            logger.info("Broadcasting synthetic data for federated validation")
            validation_results = requires.validate_synthetic_data(driver, filtered_node_ids, synthetic_csv)

            # Collect per-node reports
            utility_reports: dict = {}
            privacy_reports: dict = {}
            for node_id, reports in validation_results.items():
                utility_reports[node_id] = reports["utility_report"]
                privacy_reports[node_id] = reports["privacy_report"]
                mlflow_client.log_artifact_str(
                    reports["utility_report"],
                    f"postprocessing/reports/{node_id}_utility.html",
                    post_run_id,
                )
                mlflow_client.log_artifact_str(
                    reports["privacy_report"],
                    f"postprocessing/reports/{node_id}_privacy.html",
                    post_run_id,
                )

            # Log synthetic CSV as artifact
            mlflow_client.log_artifact_str(synthetic_csv, "postprocessing/synthetic_data.csv", post_run_id)

            # Publish to data catalogue
            import io
            synthetic_data = pd.read_csv(io.StringIO(synthetic_csv))
            publish_to_catalogue(
                synthetic_data,
                utility_reports,
                privacy_reports,
                dataset_meta,
                experiment_name=task.experiment_name,
            )

            mlflow_client.finish_run(post_run_id)
            logger.info("Post-training validation pipeline complete", post_run_id=post_run_id)

        except Exception as exc:
            logger.error("Post-training validation pipeline failed", error=str(exc))
    else:
        logger.warning(
            "dataset_meta_json not provided; skipping post-training validation",
            task_id=task.id,
        )

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
