import mlflow
from flwr.client import ClientApp
from flwr.common import Context, Message
from fl_client.src.config import MLFLOW_URL
from fl_client.src.interface import (
    get_parameters_if,
    load_data_if,
    load_model_if,
    prepare_data_if,
    set_parameters_if,
    set_run_config_if,
    train_model_if,
    upload_model_if,
)
from common.utils.src.mlflow_utils import setup_mlflow

mlflow_client = setup_mlflow(mlflow, tracking_url=MLFLOW_URL)

global_vars = {
    "data": None,
    "local_learner": None,
    "model": None,
    "model_meta": None,
    "use_case": None,
    "mlflow_experiment_id": None,
    "mlflow_run_id": None,
}

app = ClientApp()


@app.query()
def query(msg: Message, ctx: Context) -> Message:
    global global_vars

    mode = msg.content.configs_records["config"]["mode"]
    if not isinstance(mode, str):
        raise TypeError()

    if mode == "filter_clients":
        raise NotImplementedError()

    elif mode == "load_data":
        """
        Expecting a message with the following structure:
        {
            "configs_records": {
                "config": {
                    "mode": "load_data",
                    "use_case": "iris"
                }
            }
        }
        """
        return load_data_if(msg, global_vars)

    elif mode == "load_model":
        return load_model_if(msg, mlflow_client, global_vars)

    elif mode == "get_parameters":
        return get_parameters_if(msg, global_vars)

    elif mode == "set_parameters":
        return set_parameters_if(msg, global_vars)

    elif mode == "set_run_config":
        return set_run_config_if(msg, global_vars)

    elif mode == "prepare_data":
        return prepare_data_if(msg, global_vars)

    elif mode == "upload_model":
        return upload_model_if(msg, mlflow, mlflow_client, global_vars)

    else:
        raise NotImplementedError(f"Unknown mode: {mode}")


@app.train()
def train(msg: Message, ctx: Context) -> Message:
    global global_vars
    return train_model_if(msg, global_vars)


@app.evaluate()
def evaluate(msg: Message, ctx: Context) -> Message:
    raise NotImplementedError()
