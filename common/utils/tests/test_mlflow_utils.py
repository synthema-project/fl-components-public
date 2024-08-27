# import os
# import pytest
# import mlflow
# from mlflow.pyfunc import log_model
# from mlflow.entities.model_registry import ModelVersion
# from mlflow.models.model import ModelInfo
# from common.fl_models.iris.fl_model import FLModel
# from common.interfaces.src.mlflow_client import _initialized, create_parent_run, setup_mlflow, upload_final_state


# def test_setup_mlflow():
#     assert not _initialized
#     tracking_uri = "http://example.com"
#     setup_mlflow(tracking_uri)
#     assert _initialized
#     assert mlflow.get_tracking_uri() == tracking_uri

# @pytest.fixture(scope="session")
# def mlflow_client():
#     mlflow.set_tracking_uri(os.getenv("MLFLOW_URL"))
#     return mlflow.tracking.MlflowClient(tracking_uri=os.getenv("MLFLOW_URL"))

# def test_create_parent_run():
#     experiment_name = "default"
#     run_name = "test_parent_run"
#     create_parent_run(run_name, experiment_name)

# @pytest.fixture(scope="session")
# def fl_model():
#     return FLModel()


# @pytest.fixture(scope="session")
# def experiment_and_run(mlflow_client):
#     # Create an experiment and run for testing
#     try:
#         experiment_id = mlflow_client.create_experiment("test_experiment")
#     except mlflow.exceptions.MlflowException:
#         experiment = mlflow_client.get_experiment_by_name("test_experiment")
#         experiment_id = experiment.experiment_id
#         if experiment.lifecycle_stage == "deleted":
#             mlflow_client.restore_experiment(experiment_id)
#     run_id = mlflow_client.create_run(experiment_id).info.run_id
#     yield experiment_id, run_id
#     mlflow_client.delete_run(run_id)
#     mlflow_client.delete_experiment(experiment_id)


# @pytest.fixture(scope="session")
# def register_fl_model(mlflow_client, fl_model, experiment_and_run):
#     experiment_id, run_id = experiment_and_run
#     model_name = "test_name"
#     # Create a model for testing
#     with mlflow.start_run(experiment_id=experiment_id, run_id=run_id):
#         model_info: ModelInfo = log_model(
#             artifact_path="model",
#             python_model=fl_model,
#             registered_model_name=model_name,
#         )
#     model_version: ModelVersion = mlflow_client.get_model_version(
#         name=model_name, version=model_info.registered_model_version
#     )
#     yield model_info, model_version
#     mlflow_client.delete_registered_model(model_version.name)


# def test_upload_final_state(
#     experiment_and_run, mlflow_client, register_fl_model, fl_model
# ):
#     local_learner = fl_model.create_local_learner()
#     model_info, model_version = register_fl_model
#     # Set up the necessary inputs for the function
#     _, run_id = experiment_and_run
#     upload_final_state(mlflow, local_learner, model_version, run_id)

#     # Perform assertions on the output or check if the model is logged in MLflow
#     assert model_info is not None
#     assert mlflow_client.get_registered_model(model_version.name) is not None
#     assert (
#         mlflow_client.get_model_version(model_version.name, model_version.version)
#         is not None
#     )


# def test_register_model_metadata(register_fl_model, mlflow_client):
#     _, model_version = register_fl_model
#     # Add new tag so that it is available afterwards
#     model_version._add_tag(type("tag", (), {"key": "use_case", "value": "test"})())
#     register_model_metadata(mlflow_client, model_version)
#     # Perform assertions to check if the model metadata is updated correctly
#     meta = mlflow_client.get_latest_versions(f"trained_{model_version.name}")
#     assert meta[0].description == model_version.description
#     assert meta[0].tags["use_case"] == model_version.tags["use_case"]
#     assert meta[0].tags["trained"]
