from unittest.mock import Mock
from fl_server.interfaces import mlflow_client


def test_server_provisions_before_loading_model(monkeypatch):
    monkeypatch.setattr(mlflow_client._initialized, 'value', True)
    monkeypatch.setattr(mlflow_client._configured, 'value', True)
    monkeypatch.setattr(mlflow_client._current_config, 'model_name', 'demo')
    monkeypatch.setattr(mlflow_client._current_config, 'model_version', 1)
    events = []
    monkeypatch.setattr('mlflow.artifacts.download_artifacts',
                        Mock(return_value='/tmp/server-model'))
    monkeypatch.setattr('interfaces.model_environment.install_model_dependencies',
                        lambda path: events.append(('install', path)))
    monkeypatch.setattr(mlflow_client, '_load_model',
                        lambda path: events.append(('load', path)))
    mlflow_client.load_model()
    assert events == [('install', '/tmp/server-model'), ('load', '/tmp/server-model')]
