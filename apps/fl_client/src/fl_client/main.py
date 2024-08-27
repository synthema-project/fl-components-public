from flwr.client import ClientApp
from flwr.common import Context, Message

from fl_client.config import MLFLOW_URL, NODE_NAME, DATA_PATH
from fl_client.interface import (
    clean_config_if,
    filter_clients_if,
    get_parameters_if,
    load_data_if,
    load_model_if,
    prepare_data_if,
    set_parameters_if,
    set_run_config_if,
    train_model_if,
    upload_model_if,
)

from interfaces.mlflow_client import setup_mlflow

global_vars = {
    "data": None,
    "local_learner": None,
    "model": None,
    "model_meta": None,
    "use_case": "iris",
    "node_name": NODE_NAME,
    "data_path": DATA_PATH,
}

setup_mlflow(MLFLOW_URL)
app = ClientApp()


@app.query()
def query(msg: Message, ctx: Context) -> Message:
    global global_vars

    mode = msg.content.configs_records["config"]["mode"]
    if not isinstance(mode, str):
        raise TypeError()

    if mode == "filter_clients":
        return filter_clients_if(msg, global_vars)

    elif mode == "load_data":
        return load_data_if(msg, global_vars)

    elif mode == "load_model":
        return load_model_if(msg, global_vars)

    elif mode == "get_parameters":
        return get_parameters_if(msg, global_vars)

    elif mode == "set_parameters":
        return set_parameters_if(msg, global_vars)

    elif mode == "set_run_config":
        return set_run_config_if(msg, global_vars)

    elif mode == "prepare_data":
        return prepare_data_if(msg, global_vars)

    elif mode == "upload_model":
        return upload_model_if(msg, global_vars)

    elif mode == "clean_config":
        return clean_config_if(msg)

    else:
        raise NotImplementedError(f"Unknown mode: {mode}")


@app.train()
def train(msg: Message, ctx: Context) -> Message:
    global global_vars
    return train_model_if(msg, global_vars)


@app.evaluate()
def evaluate(msg: Message, ctx: Context) -> Message:
    raise NotImplementedError()
