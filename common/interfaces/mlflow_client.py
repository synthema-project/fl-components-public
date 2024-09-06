from typing import Any, Self, TypedDict, cast

import mlflow
from mlflow import MlflowClient
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PyFuncModel, load_model as _load_model
from mlflow.pytorch import log_model as _log_model
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.data.meta_dataset import MetaDataset
from mlflow.data.http_dataset_source import HTTPDatasetSource

from .utils import ensure_bool, MutableBoolean


class ModelView(TypedDict):
    name: str
    version: int
    model_id: str
    run_id: str
    uri: str


class __Config:
    experiment_id: str = cast(str, None)
    parent_run_id: str = cast(str, None)
    child_run_id: str = cast(str, None)
    model_name: str = cast(str, None)
    model_version: int = cast(int, None)
    model_tags: dict = dict()
    model_python: Any = None
    model_description: str = cast(str, None)
    is_central_node: bool = cast(bool, None)

    def __new__(cls) -> Self:
        raise RuntimeError("Cannot instantiate Config class")

    @classmethod
    def clean(cls) -> None:
        for attr in cls.__dict__:
            if not attr.startswith("__"):
                setattr(cls, attr, None)


_initialized = MutableBoolean(False)
_configured = MutableBoolean(False)
_current_config = __Config
_mlflow_client = cast(MlflowClient, None)


def setup_mlflow(tracking_url: str, is_central_node: bool = False) -> None:
    global _mlflow_client, _initialized, _current_config
    _current_config.is_central_node = is_central_node
    mlflow.set_tracking_uri(tracking_url)
    _mlflow_client = MlflowClient(tracking_url)
    _initialized.value = True


@ensure_bool(_initialized)
def create_parent_run(run_name: str, experiment_name: str) -> tuple[str, str]:
    experiment = _mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name} not found")
    experiment_id = experiment.experiment_id
    run = _mlflow_client.create_run(experiment_id=experiment_id, run_name=run_name)
    return run.info.run_id, run.info.experiment_id


@ensure_bool(_initialized)
def create_child_run(experiment_id: str, parent_run_id: str, node_name: str) -> str:
    if _mlflow_client is None:
        raise RuntimeError("Mlflow not initialized")
    run = _mlflow_client.create_run(
        experiment_id=experiment_id,
        run_name=node_name,
        tags={MLFLOW_PARENT_RUN_ID: parent_run_id},
    )
    return str(run.info.run_id)


@ensure_bool(_configured, False)
@ensure_bool(_initialized)
def set_current_config(
    experiment_id: str,
    parent_run_id: str,
    child_run_id: str,
    model_name: str,
    model_version: int,
) -> None:
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
    _current_config.model_description = cast(str, model_mlflow_meta.description)
    _configured.value = True


@ensure_bool(_configured)
def clean_current_config() -> None:
    global _current_config, _configured
    # _current_config.clean()
    if _current_config.is_central_node:
        _mlflow_client.update_run(_current_config.parent_run_id, "FINISHED")
    _mlflow_client.update_run(_current_config.child_run_id, "FINISHED")
    _configured.value = False


@ensure_bool(_configured)
@ensure_bool(_initialized)
def load_model() -> PyFuncModel:
    return _load_model(
        f"models:/{_current_config.model_name}/{_current_config.model_version}"
    )


@ensure_bool(_configured)
@ensure_bool(_initialized)
def upload_final_state(
    local_learner: Any,
) -> ModelView:
    name = f"trained_{_current_config.model_name}"
    with mlflow.start_run(
        run_id=_current_config.parent_run_id,
    ):
        model_info: ModelInfo = _log_model(
            pytorch_model=local_learner,
            artifact_path="model",
            registered_model_name=name,
        )

    version = model_info.registered_model_version
    if not isinstance(version, int):
        raise TypeError(f"Expected int, got {type(version)}")

    _mlflow_client.update_model_version(
        name, str(version), _current_config.model_description
    )
    _mlflow_client.set_model_version_tag(
        name,
        str(version),
        "use_case",
        _current_config.model_tags["use_case"],
    )
    _mlflow_client.set_model_version_tag(name, str(version), "trained", True)
    return ModelView(
        {
            "name": name,
            "version": version,
            "model_id": model_info.model_uuid,
            "run_id": model_info.run_id,
            "uri": model_info.model_uri,
        }
    )


@ensure_bool(_configured)
@ensure_bool(_initialized)
def log_metrics(
    metrics: dict[str, float],
    step: int,
) -> None:
    with mlflow.start_run(run_id=_current_config.child_run_id):
        mlflow.log_metrics(metrics, step=step)


@ensure_bool(_configured)
@ensure_bool(_initialized)
def set_dataset_signature(node_name: str, path: str, run_id: str) -> None:
    source = HTTPDatasetSource(f"{node_name}://{path}")
    dataset = MetaDataset(source)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_input(dataset, "training")
