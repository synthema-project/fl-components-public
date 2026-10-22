import os
import time
from typing import Any, Optional, Self, cast

import mlflow
import requests
import structlog
from mlflow import MlflowClient
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.data.meta_dataset import MetaDataset
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import (
    PyFuncModel,
    PythonModel,
    load_model as _load_model,
    log_model as _log_model,
)
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from .utils import ensure_bool, MutableBoolean

logger = structlog.get_logger(__name__)


class _TrainedFederatedModel(PythonModel):
    """Generic pyfunc wrapper for a trained local learner.

    Logs the final trained model the same (framework-agnostic, cloudpickled)
    way the untrained FederatedModel is registered, instead of a
    framework-specific flavor. This lets any learner — a scikit GMM, a torch
    module, etc. — be persisted without ``mlflow.pytorch`` demanding a
    ``torch.nn.Module``.
    """

    def __init__(self, local_learner: Any) -> None:
        self.local_learner = local_learner

    def predict(self, context: Any, model_input: Any = None, params: Any = None) -> Any:
        n_samples = 1
        if isinstance(params, dict) and params.get("n_samples") is not None:
            n_samples = int(params["n_samples"])
        elif model_input is not None:
            try:
                n_samples = int(len(model_input))
            except TypeError:
                pass
        generate = getattr(self.local_learner, "generate_synthetic_data", None)
        if generate is None:
            raise NotImplementedError(
                "Trained learner does not expose generate_synthetic_data()."
            )
        return generate(n_samples, getattr(self.local_learner, "dataset_schema", None))

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

# --- restapi metrics proxy config -------------------------------------------
# Metrics are logged through the restapi proxy (the same component the dashboard
# reads them from) instead of writing to MLflow directly. The client
# authenticates to the proxy with a Keycloak service-account (client_credentials)
# token. All values come from the environment (wired via Helm).
_FL_RESTAPI_URL = os.getenv("FL_RESTAPI_URL")
_KEYCLOAK_SERVER_URL = os.getenv("KEYCLOAK_SERVER_URL")
_KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM_NAME")
_KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
_KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")

_service_token: Optional[str] = None
_service_token_exp: float = 0.0


def _get_service_token() -> str:
    """Return a cached Keycloak service-account access token, refreshing it
    shortly before expiry.

    Uses the ``client_credentials`` grant so the headless fl-client can
    authenticate to the restapi metrics proxy without a human user.
    """
    global _service_token, _service_token_exp
    now = time.time()
    if _service_token and now < _service_token_exp - 30:
        return _service_token
    if not (
        _KEYCLOAK_SERVER_URL
        and _KEYCLOAK_REALM
        and _KEYCLOAK_CLIENT_ID
        and _KEYCLOAK_CLIENT_SECRET
    ):
        raise RuntimeError(
            "Keycloak service-account is not configured "
            "(need KEYCLOAK_SERVER_URL, KEYCLOAK_REALM_NAME, "
            "KEYCLOAK_CLIENT_ID, KEYCLOAK_CLIENT_SECRET)."
        )
    response = requests.post(
        f"{_KEYCLOAK_SERVER_URL}/realms/{_KEYCLOAK_REALM}/protocol/openid-connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": _KEYCLOAK_CLIENT_ID,
            "client_secret": _KEYCLOAK_CLIENT_SECRET,
        },
        timeout=10,
        verify=False,
    )
    response.raise_for_status()
    body = response.json()
    _service_token = body["access_token"]
    _service_token_exp = now + float(body.get("expires_in", 300))
    return _service_token


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

    # # Fetch token
    # token = get_token()
    # os.environ["MLFLOW_TRACKING_TOKEN"] = token

    # Initialize MLflowClient with authentication
    _mlflow_client = MlflowClient(tracking_uri=tracking_url)
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

    run = mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags={"mlflow.note.content": "Federated Generative Training Run"}
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


def clean_current_config(status: str = "FINISHED") -> None:
    """End runs and release local state, even if tracking is unavailable."""
    if status not in {"FINISHED", "FAILED"}:
        raise ValueError(f"Invalid terminal run status: {status}")
    run_ids = []
    if _configured.value:
        run_ids.append(_current_config.child_run_id)
        if _current_config.is_central_node:
            run_ids.append(_current_config.parent_run_id)
    errors = []
    try:
        # Also handles a run created before set_current_config failed.
        try:
            mlflow.end_run(status=status)
        except Exception as error:
            errors.append(error)
        for run_id in run_ids:
            if run_id:
                try:
                    _mlflow_client.set_terminated(str(run_id), status=status)
                except Exception as error:
                    errors.append(error)
    finally:
        _configured.value = False
    if errors:
        raise errors[0]


@ensure_bool(_configured)
@ensure_bool(_initialized)
def load_model() -> PyFuncModel:
    """
    Loads a model from MLflow based on the current configuration.

    Returns:
        PyFuncModel: The loaded model.
    """
    from interfaces.model_environment import load_model_with_dependencies
    model: PyFuncModel = load_model_with_dependencies(
        f"models:/{_current_config.model_name}/{_current_config.model_version}", _load_model
    )
    return model


@ensure_bool(_configured)
@ensure_bool(_initialized)
def load_dataset_meta(model: PyFuncModel) -> dict | None:
    """Read MSI's bundled feature schema using the registered model artifact path."""
    import json
    from pathlib import Path
    artifacts = model.metadata.flavors.get("python_function", {}).get("artifacts", {})
    descriptor = artifacts.get("dataset_meta")
    if descriptor is None:
        return None
    path = descriptor.get("path")
    if not isinstance(path, str) or path.startswith("/") or ".." in Path(path).parts:
        raise ValueError("Invalid dataset_meta artifact path in registered model")
    uri = f"models:/{_current_config.model_name}/{_current_config.model_version}/{path}"
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=uri)
    return json.loads(Path(local_path).read_text())


@ensure_bool(_configured)
@ensure_bool(_initialized)
def upload_final_state(
        local_learner: Any,
        use_case: Optional[str] = None,
) -> dict[str, Any]:
    """
    Uploads the final model state to MLflow and sets the appropriate tags and metadata.

    Args:
        local_learner (Any): The trained model to be uploaded.
        use_case (Optional[str]): Task use case (from the dataset) used to tag the
            trained model when the base model version carries no ``use_case`` tag.

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
            python_model=_TrainedFederatedModel(local_learner),
            artifact_path="model",
            registered_model_name=name,
        )

    version = model_info.registered_model_version
    _mlflow_client.update_model_version(
        name, str(version), _current_config.model_description
    )
    # Propagate the use case onto the trained model. Prefer the base model's own
    # tag; fall back to the task use case (from the dataset) when the base model
    # was registered without one, and skip if neither is available.
    resolved_use_case = (_current_config.model_tags or {}).get("use_case") or use_case
    if resolved_use_case is not None:
        _mlflow_client.set_model_version_tag(
            name,
            str(model_info.registered_model_version),
            "use_case",
            resolved_use_case,
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
    Logs metrics for the current child run through the restapi proxy.

    Instead of writing to MLflow directly, the metrics are POSTed to the restapi
    (the same proxy the dashboard reads them from), which forwards them to
    MLflow's log-batch API. The client authenticates with a Keycloak
    service-account token. The target run is the current child run, which lives
    under the parent run the fl-server manages and finalizes when it uploads the
    model.

    Falls back to a direct MLflow write only when FL_RESTAPI_URL is unset, so
    environments that have not been reconfigured keep working.

    Args:
        metrics (dict): A dictionary containing metric names and their values.
        step (int): The training step at which the metrics are logged.
    """
    numeric: dict[str, float] = {}
    for key, value in dict(metrics).items():
        try:
            numeric[key] = float(value)
        except (TypeError, ValueError):
            logger.warning("Skipping non-numeric metric", key=key, value=value)
    if not numeric:
        return

    if not _FL_RESTAPI_URL:
        logger.warning(
            "FL_RESTAPI_URL not set; logging metrics directly to MLflow"
        )
        mlflow.log_metrics(numeric, step=step)
        return

    response = requests.post(
        f"{_FL_RESTAPI_URL}/mlflow/runs/metrics/log",
        json={
            "run_id": _current_config.child_run_id,
            "metrics": numeric,
            "step": step,
        },
        headers={"Authorization": f"Bearer {_get_service_token()}"},
        timeout=15,
    )
    response.raise_for_status()


@ensure_bool(_configured)
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
