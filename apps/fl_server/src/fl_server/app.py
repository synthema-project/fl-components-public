import asyncio
import io
from typing import cast

import pandas as pd
import structlog
from flwr.common import Context, ParametersRecord
from flwr.server import Driver, ServerApp
from model_store_interface.catalogue_hook import publish_to_catalogue
from model_store_interface.core.schemas.dataset_meta import DatasetMeta

from fl_server import requires, stages, config
from fl_server.utils import mlflow_utils
from fl_server.interfaces import mlflow_client
from fl_server.interfaces.rabbitmq_client import AsyncRabbitMQClient
from fl_server.schemas.task import TaskRead

logger = structlog.get_logger(__name__)


async def configure_mlflow(task: TaskRead, driver: Driver, node_ids: list[int]) -> None:
    """
    Configures MLflow experiment runs on the server and synchronises the
    tracking IDs and DCAT-AP Data Contracts to all clinical edge nodes.
    """
    # 1. Clean up any stale state to prevent crash-loops
    if mlflow_client.mlflow.active_run():
        mlflow_client.mlflow.end_run("FAILED")

    # 2. Create the MLOps tracking namespace on the Server
    experiment_id, parent_run_id, child_run_id = mlflow_utils.create_mlflow_runs(
        task.run_name, task.experiment_name
    )

    # 3. Broadcast the run IDs and the DCAT-AP schema to all Federated Clients
    # This guarantees the edge nodes can open their own child runs and hydrate the metadata
    requires.set_run_config(
        driver=driver,
        node_ids=node_ids,
        experiment_id=experiment_id,
        run_id=parent_run_id,
        model_name=task.model_name,
        model_version=task.model_version,
        dataset_meta_json=getattr(task, "dataset_meta_json", None)
    )


async def load_model_and_data(task: TaskRead, driver: Driver, node_ids: list[int]) -> None:
    """Loads model and prepares data for training."""
    stages.load_model(config.global_vars)
    requires.load_data(driver, node_ids, task.use_case)
    requires.load_model(driver, node_ids, task.model_name, task.model_version)
    requires.prepare_data(driver, node_ids)


async def set_SMPC_enabled(task: TaskRead, driver: Driver, node_ids: list[int]) -> None:
    """Sets SMPC enabled flag for the task."""
    stages.set_SMPC_enabled(task.SMPC_enabled, config.global_vars)
    requires.set_SMPC_enabled(driver, node_ids, task.SMPC_enabled)


async def set_dp_config(task: TaskRead, driver: Driver, node_ids: list[int]) -> None:
    """Configures local Differential Privacy boundaries."""
    stages.configure_dp(
        config.global_vars,
        task.dp_mode,
        task.noise_multiplier,
        task.clipping_norm,
        len(node_ids)
    )
    requires.set_dp_config(
        driver, node_ids, task.dp_mode, task.noise_multiplier, task.clipping_norm
    )


async def execute_training_loop(
        driver: Driver,
        node_ids: list[int],
        parameters: ParametersRecord,
        n_global_iter: int
) -> ParametersRecord:
    """
    Executes the federated training loop, incorporating both distributed
    training (DatasetPartition.TRAIN) and intra-round evaluation (DatasetPartition.VAL).
    """
    for iteration in range(n_global_iter):
        logger.info(f"Initiating Global Federated Round {iteration + 1}/{n_global_iter}")

        # 1. Distributed Training Phase
        train_results = requires.train_model(
            driver, node_ids, parameters, current_global_iter=iteration
        )

        # Aggregate the parameters from all clinical edge nodes
        aggregated_parameters = stages.aggregate_parameters(
            [r for r in train_results], config.global_vars
        )

        # Aggregate the training telemetry metrics
        aggregated_train_metrics = stages.aggregate_metrics(
            [r[5] for r in train_results], config.global_vars
        )

        # Distributed Intra-Round Evaluation Phase (Phase 2.2)
        # We test the newly aggregated parameters against the 20% VAL hold-outs
        eval_results = requires.evaluate_model(
            driver, node_ids, aggregated_parameters
        )

        # Aggregate the evaluation telemetry metrics
        aggregated_eval_metrics = stages.aggregate_metrics(
            eval_results, config.global_vars
        )

        # 2. Telemetry Synchronisation to the MLflow Control Plane
        # Safely cast and push the metrics to our PostgreSQL backend
        mlflow_client.log_metrics(cast(dict, aggregated_train_metrics), step=iteration)
        mlflow_client.log_metrics(cast(dict, aggregated_eval_metrics), step=iteration)

        # 3. Parameter Broadcast
        # Seed the network with the newly aggregated weights for the next epoch
        requires.set_parameters(driver, node_ids, aggregated_parameters)
        parameters = aggregated_parameters

    return parameters


async def post_training_finalisation(driver: Driver, node_ids: list[int], task: TaskRead) -> None:
    """Closes the federated loop by orchestrating synthetic generation and catalogue publication."""
    if not getattr(task, "dataset_meta_json", None):
        logger.warning("No DCAT-AP schema provided. Skipping Big Pool generation and validation.")
        return

    # Extract dynamic payload target or fallback to 10k to prevent legacy task failure
    target_samples = getattr(task, "n_samples_synthetic", 10000) or 10000

    logger.info(
        "Initiating Phase 4: Federated Synthetic Data Generation for AML/SCD",
        target_samples=target_samples
    )

    # 1. Instruct the Leader Node to generate the Big Pool
    leader_node = [node_ids]

    synthetic_payloads = requires.generate_synthetic_data(driver, leader_node, target_samples)
    synthetic_csv = synthetic_payloads.get(leader_node)

    if not synthetic_csv:
        logger.error("CRITICAL: Failed to generate synthetic data from the Leader Node.")
        return

    # 2. Broadcast CSV to all clients for utility/privacy validation against real hold-outs
    logger.info("Broadcasting Big Pool for Distributed Validation...")
    validation_reports = requires.validate_synthetic_data(driver, node_ids, synthetic_csv)

    # Extract utility and privacy HTML strings safely from the RPC responses
    utility_reports = {nid: reports.get("utility", "") for nid, reports in validation_reports.items()}
    privacy_reports = {nid: reports.get("privacy", "") for nid, reports in validation_reports.items()}

    # 3. Hydrate the DCAT-AP schema locally on the Server
    dataset_meta = DatasetMeta.model_validate_json(task.dataset_meta_json)
    synthetic_df = pd.read_csv(io.StringIO(synthetic_csv))

    # 4. Delegate all heavy payload storage (CSV + HTML Reports) to MinIO via the MSI SDK
    logger.info("Pushing validated Big Pool and HTML reports directly to Private Data Catalogue...")
    publish_to_catalogue(
        synthetic_data=synthetic_df,
        utility_reports=utility_reports,
        privacy_reports=privacy_reports,
        dataset_meta=dataset_meta,
        experiment_id=task.experiment_name
    )

    logger.info("Phase 4 Finalisation complete. Model is now governed and ready for inference.")


async def server_main(task: TaskRead, driver: Driver, context: Context, rabbitmq: AsyncRabbitMQClient) -> None:
    """Main asynchronous execution flow for the Federated Learning task."""
    if not getattr(task, "id", None) or not getattr(task, "use_case", None):
        logger.error("Task is missing required fields", task_id=getattr(task, "id", "Unknown"))
        return

    logger.info("Task started", task_id=task.id)

    try:
        # Wait for downstream edge nodes to connect
        node_ids = []
        while len(node_ids) == 0:
            node_ids = driver.get_node_ids()
            if len(node_ids) == 0:
                logger.info("Waiting for clients to connect...")
                await asyncio.sleep(5)

        logger.info("Clients connected", node_ids=node_ids)

        filtered_node_ids = requires.filter_clients(driver, node_ids, task.use_case)

        logger.info("Configuring MLflow and Broadcasting Data Contracts")
        await configure_mlflow(task, driver, filtered_node_ids)

        logger.info("Loading model and data")
        await load_model_and_data(task, driver, filtered_node_ids)

        if getattr(task, "SMPC_enabled", False):
            await set_SMPC_enabled(task, driver, filtered_node_ids)
            port = config.P2P_PORT
            logger.info("Starting P2P servers")
            requires.start_P2P_server(driver, filtered_node_ids, port)

            logger.info("Discover peers")
            discovered_peers = requires.discover_peers(driver, filtered_node_ids, port, max_discovery_time=60)

            logger.info("Establishing P2P network")
            requires.connect_peers(driver, filtered_node_ids, discovered_peers)

        if getattr(task, "dp_mode", None):
            logger.info(
                f"Configuring DP {task.dp_mode} with clipping norm {task.clipping_norm} and noise multiplier {task.noise_multiplier}"
            )
            await set_dp_config(task, driver, filtered_node_ids)

        logger.info("Getting initial parameters")
        parameters = requires.get_parameters_from_one_node(driver, filtered_node_ids)

        logger.info("Broadcasting parameters")
        requires.set_parameters(driver, filtered_node_ids, parameters)

        logger.info("Starting training loop")
        parameters = await execute_training_loop(
            driver=driver,
            node_ids=filtered_node_ids,
            parameters=parameters,
            n_global_iter=task.num_global_iterations
        )

        logger.info("Uploading trained model")
        # Directs the Leader Node to upload the PyFunc model weights to the registry
        requires.upload_model(driver, filtered_node_ids, parameters)

        # Execute Post-Training Verification and Data Catalogue insertion
        if getattr(task, "dataset_meta_json", None):
            logger.info("Starting post-training validation pipeline")
            await post_training_finalisation(driver, filtered_node_ids, task)

        logger.info("Federated Training and Validation successful", task_id=task.id)

    except Exception as e:
        logger.error(f"Task {getattr(task, 'id', 'Unknown')} failed catastrophically: {e}")
        raise e

    finally:
        # ✅ RESTORED MISSING STEP: The Hermetic Closure
        logger.info("Initiating FL Session Closure & MLflow state cleanup")
        try:
            # 1. Command all active Ubuntu edge nodes to close their child runs
            if 'filtered_node_ids' in locals() and len(filtered_node_ids) > 0:
                requires.clean_config(driver, filtered_node_ids)

            # 2. Close the server's parent orchestration run
            mlflow_client.clean_current_config()
            logger.info("MLflow runs successfully transitioned to FINISHED.")
        except Exception as cleanup_error:
            logger.error(f"Failed to cleanly close MLflow runs: {cleanup_error}")


def get_serverapp(task: TaskRead, rabbitmq: AsyncRabbitMQClient) -> ServerApp:
    """Returns a configured ServerApp instance with a synchronous wrapper for Flower."""
    app = ServerApp()

    @app.main()
    def main(driver: Driver, context: Context) -> None:
        # Bridge the synchronous Flower callback to our asynchronous orchestration
        asyncio.run(server_main(task, driver, context, rabbitmq))

    return app
