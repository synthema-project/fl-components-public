import mlflow

from fl_server.src.config import MLFLOW_URL
from fl_server.src.models.task import _get_dummy_task
from fl_server.src.app import get_serverapp
from common.utils.src.flower_utils.server_app import run_server_app

from common.utils.src.mlflow_utils import setup_mlflow

mlflow_client = setup_mlflow(mlflow, MLFLOW_URL)

task = _get_dummy_task()
app = get_serverapp(mlflow, mlflow_client, task)
run_server_app(app)
