import os
import cloudpickle
from mlflow.pyfunc import PythonModel
from mlflow import MlflowClient
from mlflow.models.model import ModelInfo

from . import aggregator
from . import local_learner
from . import utils
from .aggregator import create_aggregator
from .local_learner import create_local_learner

cloudpickle.register_pickle_by_value(aggregator)
cloudpickle.register_pickle_by_value(local_learner)
cloudpickle.register_pickle_by_value(utils)


class FLModel(PythonModel):
    def __init__(self):
        self.create_aggregator = create_aggregator
        self.create_local_learner = create_local_learner


if __name__ == "__main__":
    # with open("model.pkl", "wb") as f:
    #     cloudpickle.dump(FLModel(), f)

    import mlflow
    from mlflow.pyfunc import log_model

    MLFLOW_URL = os.environ["MLFLOW_URL"]

    mlflow.set_tracking_uri(MLFLOW_URL)
    model_info: ModelInfo = log_model(
        artifact_path="model",
        python_model=FLModel(),
        registered_model_name="iris_model",
    )
    mlflow_client = MlflowClient(tracking_uri=MLFLOW_URL)
    mlflow_client.set_model_version_tag(
        "iris_model", model_info.registered_model_version, "use_case", "iris"
    )
