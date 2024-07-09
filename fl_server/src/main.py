import mlflow
import flwr as fl
from flwr.common import Context
from flwr.server import Driver

from fl_server.src.config import MLFLOW_URL
from fl_server.src.requires import (
    require_get_parameters_from_one_node,
    require_load_data,
    require_load_model,
    require_prepare_data,
    require_set_parameters,
    require_set_run_config,
    require_train_model,
    require_upload_model,
)
from fl_server.src.stages import aggregate_parameters, load_model

from common.utils.src.mlflow_utils import setup_mlflow

app = fl.server.ServerApp()
mlflow_client = setup_mlflow(mlflow, MLFLOW_URL)

global_vars = {
    "aggregator": None,
    "model": None,
    "model_meta": None,
    "use_case": None,
    "mlflow_experiment_id": None,
    "mlflow_run_id": None,
}


@app.main()
def main(driver: Driver, context: Context) -> None:
    global global_vars
    # Get node IDs
    node_ids = driver.get_node_ids()

    # Filter clients
    print("Filter clients")
    # filtered_node_ids = require_filter_clients(driver, node_ids)
    filtered_node_ids = node_ids

    # Load model and data
    print("Load model and data")
    load_model("iris_model", 3, mlflow_client, global_vars)
    require_load_data(driver, filtered_node_ids, "iris")
    require_load_model(driver, filtered_node_ids, "iris_model", 3)
    require_prepare_data(driver, filtered_node_ids)

    # Get parameters from random node
    print("Get parameters")
    parameters = require_get_parameters_from_one_node(driver, filtered_node_ids)

    # Broadcast mlflow run details
    print("Broadcast mlflow run details")
    experiment = mlflow.set_experiment("iris")
    run = mlflow_client.create_run(experiment_id=experiment.experiment_id)
    global_vars["mlflow_experiment_id"] = experiment.experiment_id
    global_vars["mlflow_run_id"] = run.info.run_id
    require_set_run_config(
        driver, filtered_node_ids, experiment.experiment_id, run.info.run_id
    )

    # Broadcast parameters
    print("Broadcast parameters")
    require_set_parameters(driver, filtered_node_ids, parameters)

    # Training loop
    print("Train 1 iter")
    results = require_train_model(driver, filtered_node_ids, parameters)
    aggregated_parameters = aggregate_parameters([r[0] for r in results], global_vars)
    require_set_parameters(driver, filtered_node_ids, aggregated_parameters)

    # Upload model
    print("Upload model")
    require_upload_model(driver, filtered_node_ids, parameters)
