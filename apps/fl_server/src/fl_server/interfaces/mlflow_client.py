from typing import Any, Optional, Self, cast

import mlflow
import structlog
from mlflow import MlflowClient
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PyFuncModel, load_model as _load_model
from mlflow.pytorch import log_model as _log_model
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from .utils import ensure_bool, MutableBoolean

logger = structlog.get_logger(__name__)

mlflow.set_system_metrics_sampling_interval(0.5)


class __Config:
    """
    A configuration class for managing experiment settings and model metadata.

    Attributes:
        experiment_id (Optional[str]): The ID of the experiment.
        parent_run_id (Optional[str]): The ID of the parent run.
        child_run_id (Optional[str]): The ID of the child run.
        model_name (Optional[str]): The name of the model.
        model_version (Optional[int]): The version of the model.
        model_tags (dict): A dictionary of tags associated with the model.
        model_python (Any): Python-specific metadata related to the model.
        model_description (Optional[str]): A description of the model.
        is_central_node (Optional[bool]): Whether this is a central node in the configuration.
    """

    experiment_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    child_run_id: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[int] = None
    model_tags: dict = dict()
    model_python: Any = None
    model_description: Optional[str] = None
    is_central_node: Optional[bool] = None

    def __new__(cls) -> Self:
        """
        Prevent instantiation of the Config class.

        Raises:
            RuntimeError: Always raises an exception to prevent instantiation.
        """
        raise RuntimeError("Cannot instantiate Config class")

    @classmethod
    def clean(cls) -> None:
        """
        Resets all class attributes to None, except for special methods and properties.

        This method is intended to clear the configuration state.
        """
        for attr in cls.__dict__:
            if not attr.startswith("__"):
                setattr(cls, attr, None)


_initialized = MutableBoolean(False)
_configured = MutableBoolean(False)
_current_config = __Config
_mlflow_client = cast(MlflowClient, None)


def setup_mlflow(tracking_url: str, is_central_node: bool = False) -> None:
    """
    Sets up the MLflow tracking URI and initializes the MlflowClient.

    Args:
        tracking_url (str): The URL of the MLflow tracking server.
        is_central_node (bool, optional): Indicates if this is a central node in a multi-run experiment.
    """
    global _mlflow_client, _initialized, _current_config
    _current_config.is_central_node = is_central_node
    mlflow.set_tracking_uri(tracking_url)
    _mlflow_client = MlflowClient(tracking_url)
    _initialized.value = True


@ensure_bool(_initialized)
def create_parent_run(run_name: str, experiment_name: str) -> tuple[str, str]:
    """
    Creates a parent run in the MLflow experiment.
    Mathematically ensures idempotency by verifying experiment existence prior to creation,
    preventing crash-loops on uninitialised Ubuntu edge nodes.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        logger.info(f"Experiment '{experiment_name}' not found. Auto-creating MLOps namespace...")
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    run = _mlflow_client.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags={"mlflow.note.content": "Federated Generative Training Run"},
    )
    return run.info.run_id, experiment_id


@ensure_bool(_initialized)
def create_child_run(experiment_id: str, parent_run_id: str, node_name: str) -> str:
    """
    Creates a child run under the specified parent run in the MLflow experiment.

    Args:
        experiment_id (str): The experiment ID in which the child run is created.
        parent_run_id (str): The ID of the parent run.
        node_name (str): The name of the child node.

    Returns:
        str: The run ID of the created child run.
    """
    if _mlflow_client is None:
        raise RuntimeError("Mlflow not initialized")
    run = _mlflow_client.create_run(
        experiment_id=experiment_id,
        run_name=node_name,
        tags={MLFLOW_PARENT_RUN_ID: parent_run_id},
    )
    mlflow.start_run(
        experiment_id=experiment_id, run_id=run.info.run_id, log_system_metrics=True
    )
    return str(run.info.run_id)


@ensure_bool(_configured, False)
@ensure_bool(_initialized)
def set_current_config(
        experiment_id: str,
        parent_run_id: str | None,
        child_run_id: str,
        model_name: str,
        model_version: int,
) -> None:
    """
    Sets the current configuration with experiment and model metadata.

    Args:
        experiment_id (str): The ID of the experiment.
        parent_run_id (str | None): The ID of the parent run (if applicable).
        child_run_id (str): The ID of the child run.
        model_name (str): The name of the model.
        model_version (int): The version of the model.
    """
    global _current_config, _mlflow_client
    model_mlflow_meta = _mlflow_client.get_model_version(
        name=model_name, version=str(model_version)
    )
    _current_config.experiment_id = experiment_id
    _current_config.parent_run_id = parent_run_id
    _current_config.child_run_id = child_run_id
    _current_config.model_name = model_name
    _current_config.model_version = model_version
    _current_config.model_tags = model_mlflow_meta.tags
    _current_config.model_description = model_mlflow_meta.description
    _configured.value = True


@ensure_bool(_configured)
def clean_current_config() -> None:
    """
    Cleans up the current configuration by finishing the associated MLflow runs.

    If the node is a central node, it updates the parent run and finishes the child run.
    """
    global _current_config, _configured
    # _current_config.clean()
    if _current_config.is_central_node:
        _mlflow_client.update_run(str(_current_config.parent_run_id), "FINISHED")
    mlflow.end_run()
    _mlflow_client.update_run(str(_current_config.child_run_id), "FINISHED")
    _configured.value = False


@ensure_bool(_configured)
@ensure_bool(_initialized)
def load_model() -> PyFuncModel:
    """
    Loads a model from MLflow based on the current configuration.

    Returns:
        PyFuncModel: The loaded model.
    """
    model: PyFuncModel = _load_model(
        f"models:/{_current_config.model_name}/{_current_config.model_version}"
    )
    return model


@ensure_bool(_configured)
@ensure_bool(_initialized)
def upload_final_state(
        local_learner: Any,
) -> dict[str, Any]:
    """
    Uploads the final model state to MLflow and sets the appropriate tags and metadata.

    Args:
        local_learner (Any): The trained model to be uploaded.

    Returns:
        dict: A dictionary containing information about the registered model.
    """
    name = f"trained_{_current_config.model_name}"
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run(
            run_id=_current_config.parent_run_id,
    ):
        model_info: ModelInfo = _log_model(
            pytorch_model=local_learner,
            artifact_path="model",
            registered_model_name=name,
        )

    version = model_info.registered_model_version
    _mlflow_client.update_model_version(
        name, str(version), _current_config.model_description
    )
    _mlflow_client.set_model_version_tag(
        name,
        str(model_info.registered_model_version),
        "use_case",
        _current_config.model_tags["use_case"],
    )
    _mlflow_client.set_model_version_tag(
        name, str(model_info.registered_model_version), "trained", True
    )
    return {
        "name": name,
        "version": version,
        "model_id": model_info.model_uuid,
        "run_id": model_info.run_id,
        "uri": model_info.model_uri,
    }


@ensure_bool(_configured)
@ensure_bool(_initialized)
def log_metrics(
        metrics: dict,
        step: int,
) -> None:
    """
    Logs metrics to MLflow for the current child run.

    Args:
        metrics (dict): A dictionary containing metric names and their values.
        step (int): The training step at which the metrics are logged.
    """
    mlflow.log_metrics(metrics, step=step)


@ensure_bool(_initialized)
def set_dataset_signature(node_name: str, path: str, run_id: str) -> None:
    """
    Logs the dataset signature used for training a model to MLflow.

    Args:
        node_name (str): The name of the node associated with the dataset.
        path (str): The path to the dataset source.
        run_id (str): The run ID under which the dataset is logged.
    """
    source = HTTPDatasetSource(f"{node_name}://{path}")
    dataset = MetaDataset(source)  # type: ignore
    mlflow.log_input(dataset, "training")


@ensure_bool(_initialized)
def create_postprocessing_run(experiment_id: str, parent_run_id: str) -> str:
    """Create a sibling MLflow run tagged as 'postprocessing'.

    Args:
        experiment_id (str): The experiment that owns the parent training run.
        parent_run_id (str): The ID of the parent federated training run.

    Returns:
        str: The run ID of the newly created postprocessing run.
    """
    run = _mlflow_client.create_run(
        experiment_id=experiment_id,
        run_name="postprocessing",
        tags={
            MLFLOW_PARENT_RUN_ID: parent_run_id,
            "type": "postprocessing",
        },
    )
    return str(run.info.run_id)


@ensure_bool(_initialized)
def log_artifact_str(content: str, artifact_path: str, run_id: str) -> None:
    """Log a string payload as an MLflow artifact on a specific run.

    Writes *content* to a temporary file and uploads it under *artifact_path*
    inside the given *run_id*'s artifact store.  The temporary file is always
    removed afterwards.

    Args:
        content (str): Text content to upload.
        artifact_path (str): Destination path within the run's artifact store
            (e.g. ``"postprocessing/reports/node1_utility.html"``).
        run_id (str): Target MLflow run ID.
    """
    import tempfile
    import os

    # Split artifact_path into directory and filename
    artifact_dir = "/".join(artifact_path.split("/")[:-1]) if "/" in artifact_path else None
    filename = artifact_path.split("/")[-1]

    with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        _mlflow_client.log_artifact(run_id, tmp_path, artifact_path=artifact_dir)
    finally:
        os.unlink(tmp_path)


def finish_run(run_id: str) -> None:
    """Mark an MLflow run as FINISHED.

    Args:
        run_id (str): The run ID to finish.
    """
    _mlflow_client.update_run(run_id, "FINISHED")
