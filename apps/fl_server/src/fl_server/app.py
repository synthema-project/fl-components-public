from flwr.common import Context, ParametersRecord
from flwr.server import Driver, ServerApp

from fl_server.requires import (
    require_clean_config,
    require_filter_clients,
    require_get_parameters_from_one_node,
    require_load_data,
    require_load_model,
    require_prepare_data,
    require_set_parameters,
    require_set_run_config,
    require_train_model,
    require_upload_model,
)
from fl_server.stages import aggregate_metrics, aggregate_parameters, load_model
from fl_server.config import global_vars
from fl_server.utils.mlflow_utils import create_mlflow_runs

from interfaces import mlflow_client
from schemas.task import Task


def _mlflow_config(task: Task, driver: Driver, node_ids: list[int]) -> None:
    experiment_id, parent_run_id, child_run_id = create_mlflow_runs(
        task.run_name, task.experiment_name
    )
    mlflow_client.set_current_config(
        experiment_id, parent_run_id, child_run_id, task.model_name, task.model_version
    )
    require_set_run_config(
        driver,
        node_ids,
        experiment_id,
        parent_run_id,
        task.model_name,
        task.model_version,
    )


def _model_and_data(task: Task, driver: Driver, node_ids: list[int]) -> None:
    # Load model and data
    load_model(global_vars)
    require_load_data(driver, node_ids, task.use_case)
    require_load_model(driver, node_ids, task.model_name, task.model_version)
    require_prepare_data(driver, node_ids)


def _training_loop(
    driver: Driver,
    node_ids: list[int],
    parameters: ParametersRecord,
    n_global_iter: int,
) -> None:
    for iter in range(n_global_iter):
        results = require_train_model(
            driver, node_ids, parameters, current_global_iter=iter
        )
        aggregated_parameters = aggregate_parameters(
            [r[0] for r in results], global_vars
        )
        aggregated_metrics = aggregate_metrics([r[1] for r in results], global_vars)
        mlflow_client.log_metrics(aggregated_metrics, step=iter)
        require_set_parameters(driver, node_ids, aggregated_parameters)


def get_serverapp(task: Task) -> ServerApp:
    def server_main(driver: Driver, context: Context) -> None:
        global global_vars
        # Get node IDs
        node_ids = driver.get_node_ids()

        # Filter clients
        print("Filter clients")
        filtered_node_ids = require_filter_clients(driver, node_ids, task.use_case)

        # Mlflow config
        print("Broadcast mlflow run details")
        _mlflow_config(task, driver, filtered_node_ids)

        # Load model and data
        print("Load model and data")
        _model_and_data(task, driver, filtered_node_ids)

        # Get parameters from random node
        print("Get parameters")
        parameters = require_get_parameters_from_one_node(driver, filtered_node_ids)

        # Broadcast parameters
        print("Broadcast parameters")
        require_set_parameters(driver, filtered_node_ids, parameters)

        # Training loop
        print("Training loop")
        _training_loop(
            driver, filtered_node_ids, parameters, task.num_global_iterations
        )

        # Upload model
        print("Upload model")
        require_upload_model(driver, filtered_node_ids, parameters)

        # Clean run details
        print("Clean run details")
        mlflow_client.clean_current_config()
        require_clean_config(driver, filtered_node_ids)

    app = ServerApp()
    app._main = server_main
    return app
