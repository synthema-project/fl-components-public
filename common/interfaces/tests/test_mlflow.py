from unittest.mock import MagicMock, Mock, patch
import pytest
from interfaces import mlflow_client

pytest.fixture(autouse=True)


def reset_mlflow_client():
    mlflow_client._initialized.value = False
    mlflow_client._configured.value = False
    mlflow_client._current_config.clean()
    mlflow_client._mlflow_client = None


def test_setup_mlflow():
    assert not mlflow_client._initialized
    tracking_uri = "http://example.com"
    mlflow_client.setup_mlflow(tracking_uri)
    assert mlflow_client._initialized
    assert mlflow_client.mlflow.get_tracking_uri() == tracking_uri


def test_create_parent_run():
    mlflow_client._initialized.value = True
    experiment_name = "test_experiment"
    experiment_id = "test_experiment_id"
    run_name = "test_run"
    run_id = "test_run_id"

    with patch(
        "interfaces.mlflow_client._mlflow_client.get_experiment_by_name"
    ) as mock_get_experiment_by_name:
        with patch(
            "interfaces.mlflow_client._mlflow_client.create_run"
        ) as mock_create_run:
            mock_get_experiment_by_name.return_value = Mock(
                experiment_id="test_experiment_id"
            )

            info = Mock(experiment_id=experiment_id, run_id=run_id)
            mock_create_run.return_value = Mock(info=info)

            run_id, experiment_id = mlflow_client.create_parent_run(
                run_name, experiment_name
            )

    mock_get_experiment_by_name.assert_called_once_with(experiment_name)
    mock_create_run.assert_called_once_with(
        experiment_id=experiment_id, run_name=run_name
    )

    assert run_id == "test_run_id"
    assert experiment_id == "test_experiment_id"


def test_create_child_run():
    mlflow_client._initialized.value = True
    experiment_id = "test_experiment_id"
    parent_run_id = "test_parent_run_id"
    node_name = "test_node_name"
    run_id = "test_run_id"

    with patch("interfaces.mlflow_client._mlflow_client.create_run") as mock_create_run:
        info = Mock(run_id=run_id)
        mock_create_run.return_value = Mock(info=info)

        run_id = mlflow_client.create_child_run(experiment_id, parent_run_id, node_name)

    mock_create_run.assert_called_once_with(
        experiment_id=experiment_id,
        run_name=node_name,
        tags={mlflow_client.MLFLOW_PARENT_RUN_ID: parent_run_id},
    )

    assert run_id == "test_run_id"


def test_set_current_config():
    experiment_id = "test_experiment_id"
    parent_run_id = "test_parent_run_id"
    child_run_id = "test_child_run_id"
    model_name = "test_model_name"
    model_version = 1
    tags = {"tag1": "value1"}
    description = "description"

    mlflow_client._initialized.value = True
    mlflow_client._configured.value = False

    with patch(
        "interfaces.mlflow_client._mlflow_client.get_model_version"
    ) as mock_get_model_version:
        mock_get_model_version.return_value = Mock(tags=tags, description=description)

        mlflow_client.set_current_config(
            experiment_id, parent_run_id, child_run_id, model_name, model_version
        )

    assert mlflow_client._current_config.experiment_id == experiment_id
    assert mlflow_client._current_config.parent_run_id == parent_run_id
    assert mlflow_client._current_config.child_run_id == child_run_id
    assert mlflow_client._current_config.model_name == model_name
    assert mlflow_client._current_config.model_version == model_version
    assert mlflow_client._current_config.model_tags == tags
    assert mlflow_client._current_config.model_description == description
    assert mlflow_client._configured


def test_clean_current_config():
    mlflow_client._initialized.value = True
    mlflow_client._configured.value = True

    mlflow_client.clean_current_config()

    assert not mlflow_client._configured
    assert mlflow_client._current_config.experiment_id is None
    assert mlflow_client._current_config.parent_run_id is None
    assert mlflow_client._current_config.child_run_id is None
    assert mlflow_client._current_config.model_name is None
    assert mlflow_client._current_config.model_version is None
    assert mlflow_client._current_config.model_tags is None
    assert mlflow_client._current_config.model_description is None


def test_load_mlflow_model():
    mlflow_client._initialized.value = True
    mlflow_client._configured.value = True

    mlflow_client._current_config.model_name = "test_model_name"
    mlflow_client._current_config.model_version = 1

    with patch("interfaces.mlflow_client._load_model") as mock_load_model:
        mlflow_client.load_model()

    mock_load_model.assert_called_once_with(
        f"models:/{mlflow_client._current_config.model_name}/{mlflow_client._current_config.model_version}"
    )


def test_upload_final_state():
    model_name = "test_model_name"
    uploaded_model_name = f"trained_{model_name}"
    model_version = 1
    description = "description"
    tags = {"use_case": "test_use_case"}
    model_id = "test_model_uuid"
    run_id = "test_run_id"
    model_uri = "test_model_uri"

    mlflow_client._initialized.value = True
    mlflow_client._configured.value = True
    mlflow_client._current_config.model_name = model_name
    mlflow_client._current_config.model_version = model_version
    mlflow_client._current_config.model_description = description
    mlflow_client._current_config.model_tags = tags

    final_state = Mock()

    with patch("interfaces.mlflow_client.mlflow") as mock_mlflow:
        mock_mlflow.start_run = MagicMock()
        with patch("interfaces.mlflow_client._log_model") as mock_log_model:
            model_info = type(
                "ModelInfo",
                (),
                {
                    "registered_model_version": model_version,
                    "model_uuid": model_id,
                    "run_id": run_id,
                    "model_uri": model_uri,
                },
            )()
            mock_log_model.return_value = model_info
            with patch("interfaces.mlflow_client._mlflow_client") as mock_mlflow_client:
                mock_mlflow_client.update_model_version = Mock()
                mock_mlflow_client.set_model_version_tag = Mock()

                result = mlflow_client.upload_final_state(final_state)

    mock_log_model.assert_called_once_with(
        pytorch_model=final_state,
        artifact_path="model",
        registered_model_name=uploaded_model_name,
    )

    mock_mlflow_client.update_model_version.assert_called_once_with(
        uploaded_model_name, model_version, description
    )
    mock_mlflow_client.set_model_version_tag.assert_any_call(
        uploaded_model_name, model_version, "use_case", tags["use_case"]
    )
    mock_mlflow_client.set_model_version_tag.assert_any_call(
        uploaded_model_name, model_version, "trained", True
    )

    assert result == {
        "name": uploaded_model_name,
        "version": model_version,
        "model_id": model_id,
        "run_id": run_id,
        "uri": model_uri,
    }


def test_log_metrics():
    mlflow_client._initialized.value = True
    mlflow_client._configured.value = True

    metrics = {"metric1": 1}
    step = 1

    with patch("interfaces.mlflow_client.mlflow") as mock_mlflow:
        mock_mlflow.start_run = MagicMock()
        mlflow_client.log_metrics(metrics, step)

    mock_mlflow.log_metrics.assert_called_once_with(metrics, step=step)


def test_set_dataset_signature():
    mlflow_client._initialized.value = True
    mlflow_client._configured.value = True

    with patch("interfaces.mlflow_client.mlflow") as mock_mlflow:
        mock_mlflow.start_run = MagicMock()
        mock_mlflow.log_input = Mock()
        mlflow_client.set_dataset_signature(
            "test_node_name",
            "test_path",
            "test_run_id",
        )

    mock_mlflow.log_input.assert_called_once()
