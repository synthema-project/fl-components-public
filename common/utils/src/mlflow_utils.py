import mlflow
from mlflow import MlflowClient
from mlflow.models.model import ModelInfo
from mlflow.entities.model_registry import ModelVersion
from mlflow.pyfunc import PyFuncModel, load_model
from torch.nn import Module


def setup_mlflow(mlflow_module: mlflow, tracking_url: str) -> MlflowClient:
    mlflow_module.set_tracking_uri(tracking_url)
    return MlflowClient(tracking_url)


def load_mlflow_model(
    mlflow_client: mlflow.MlflowClient,
    model_name: str,
    model_version: int,
) -> tuple[PyFuncModel, ModelVersion]:
    mlflow_model: PyFuncModel = load_model(f"models:/{model_name}/{model_version}")
    model_meta = mlflow_client.get_model_version(name=model_name, version=model_version)
    return mlflow_model, model_meta


def upload_final_state(
    mlflow: mlflow,
    local_learner: Module,
    model_meta: ModelVersion,
    run_id: str,
) -> ModelInfo:
    with mlflow.start_run(
        run_id=run_id,
    ):
        model_info: mlflow.models.model.ModelInfo = mlflow.pytorch.log_model(
            pytorch_model=local_learner,
            artifact_path="model",
            registered_model_name=f"trained_{model_meta.name}",
        )

    return model_info


def register_model_metadata(
    mlflow_client: MlflowClient, model_meta: ModelVersion
) -> None:
    meta = mlflow_client.get_latest_versions(f"trained_{model_meta.name}")
    mlflow_client.update_model_version(
        meta[0].name, meta[0].version, model_meta.description
    )

    mlflow_client.set_model_version_tag(
        meta[0].name, meta[0].version, "use_case", model_meta.tags["use_case"]
    )
    mlflow_client.set_model_version_tag(meta[0].name, meta[0].version, "trained", True)


def log_metrics(
    mlflow_module: mlflow,
    metrics: dict,
    step: int,
    run_id: str,
) -> None:
    with mlflow_module.start_run(run_id=run_id):
        mlflow_module.log_metrics(metrics, step=step)
