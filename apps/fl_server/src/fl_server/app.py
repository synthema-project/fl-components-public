from typing import cast
from flwr.common import Context, ParametersRecord
from flwr.server import Driver, ServerApp

from fl_server import requires
from fl_server import stages
from fl_server import config
from fl_server.utils import mlflow_utils

from interfaces import mlflow_client
from schemas.task import Task


def _mlflow_config(task: Task, driver: Driver, node_ids: list[int]) -> None:
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


def _model_and_data(task: Task, driver: Driver, node_ids: list[int]) -> None:
    # Load model and data
    stages.load_model(config.global_vars)
    requires.load_data(driver, node_ids, task.use_case)
    requires.load_model(driver, node_ids, task.model_name, task.model_version)
    requires.prepare_data(driver, node_ids)


def _training_loop(
    driver: Driver,
    node_ids: list[int],
    parameters: ParametersRecord,
    n_global_iter: int,
) -> None:
    for iter in range(n_global_iter):
        results = requires.train_model(
            driver, node_ids, parameters, current_global_iter=iter
        )
        aggregated_parameters = stages.aggregate_parameters(
            [r[0] for r in results], config.global_vars
        )
        aggregated_metrics = stages.aggregate_metrics(
            [r[1] for r in results], config.global_vars
        )
        mlflow_client.log_metrics(cast(dict[str, float], aggregated_metrics), step=iter)
        requires.set_parameters(driver, node_ids, aggregated_parameters)


def get_serverapp(task: Task) -> ServerApp:
    def server_main(driver: Driver, context: Context) -> None:
        global global_vars
        # Get node IDs
        node_ids = driver.get_node_ids()

        # Filter clients
        print("Filter clients")
        filtered_node_ids = requires.filter_clients(driver, node_ids, task.use_case)

        # Mlflow config
        print("Broadcast mlflow run details")
        _mlflow_config(task, driver, filtered_node_ids)

        # Load model and data
        print("Load model and data")
        _model_and_data(task, driver, filtered_node_ids)

        # Get parameters from random node
        print("Get parameters")
        parameters = requires.get_parameters_from_one_node(driver, filtered_node_ids)

        # Broadcast parameters
        print("Broadcast parameters")
        requires.set_parameters(driver, filtered_node_ids, parameters)

        # Training loop
        print("Training loop")
        _training_loop(
            driver, filtered_node_ids, parameters, task.num_global_iterations
        )

        # Upload model
        print("Upload model")
        requires.upload_model(driver, filtered_node_ids, parameters)

        # Clean run details
        print("Clean run details")
        mlflow_client.clean_current_config()
        requires.clean_config(driver, filtered_node_ids)

    app = ServerApp()
    app._main = server_main
    return app
