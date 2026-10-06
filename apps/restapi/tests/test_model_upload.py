import io
import json
from pathlib import Path
import tarfile
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import yaml

from restapi.auth import get_mock_user, require_authentication
from restapi.helpers import model_upload
from restapi.routers import models


def files():
    descriptor = {'model_uuid': 'original-model-uuid', 'flavors': {'python_function': {
        'loader_module': 'mlflow.pyfunc.model', 'python_model': 'python_model.pkl',
        'code': 'code', 'env': {'conda': 'conda.yaml', 'virtualenv': 'python_env.yaml'},
        'artifacts': {'dataset_meta': {'path': 'artifacts/dataset_meta.json'},
                      'license': {'path': 'artifacts/LICENSE.txt'}}}}}
    return {'MLmodel': yaml.safe_dump(descriptor), 'python_model.pkl': 'opaque pickle bytes',
            'conda.yaml': '{}', 'python_env.yaml': '{}', 'requirements.txt': 'numpy==1.26.4',
            'code/example.py': "raise AssertionError('Do not import in the API')",
            'artifacts/dataset_meta.json': json.dumps({'use_case': 'AML', 'variables': [{'name': 'Age'}]}),
            'artifacts/LICENSE.txt': 'MIT', 'evaluation_config.json': '{}'}


def archive(contents=None, special=None, prefix=''):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for name, content in (contents or files()).items():
            raw = content.encode() if isinstance(content, str) else content
            member = tarfile.TarInfo(prefix + name)
            member.size = len(raw)
            tar.addfile(member, io.BytesIO(raw))
        if special:
            tar.addfile(special)
    return stream.getvalue()


def client(authenticated=True):
    app = FastAPI()
    app.include_router(models.router)
    if authenticated:
        app.dependency_overrides[require_authentication] = get_mock_user
    return TestClient(app)


def post(c, bundle, **changes):
    return c.post('/api/v1/models/upload',
        data={'experiment_name': 'test', 'use_case': 'AML', 'trained': 'false',
              'model_name': 'demo', 'aggregator_name': 'AVG', **changes},
        files={'bundle': ('model.tar.gz', bundle, 'application/gzip')})


@pytest.mark.parametrize('prefix', ['', 'model/', './'])
def test_upload_contract_and_temp_cleanup(monkeypatch, prefix):
    saved = []
    def register(folder, experiment, disease, trained, name, aggregator, user_id, descriptor):
        assert experiment == 'test' and disease == 'AML' and trained is False
        assert name == 'demo' and aggregator == 'AVG' and user_id == 'test-user'
        assert descriptor['model_uuid'] == 'original-model-uuid'
        assert (folder / 'python_model.pkl').read_bytes() == b'opaque pickle bytes'
        saved.append(folder)
        return {'message': 'Model uploaded and registered', 'model_id': 'original-model-uuid',
                'model_uri': 'models:/demo/1', 'run_id': 'run', 'experiment_id': '1', 'model_version': '1'}
    monkeypatch.setattr(models, 'register_bundle', register)
    response = post(client(), archive(prefix=prefix))
    assert response.status_code == 201, response.text
    assert response.json()['model_uri'] == 'models:/demo/1'
    assert not saved[0].exists()


def test_requires_authentication(monkeypatch):
    monkeypatch.setattr('restapi.auth.AUTH_DISABLE_TESTS', False)
    assert post(client(False), archive()).status_code == 401


@pytest.mark.parametrize('name,kind', [('../escape', 'file'), ('/escape', 'file'),
    ('link', 'symlink'), ('hardlink', 'hardlink'), ('pipe', 'fifo'), ('MLmodel', 'file')])
def test_unsafe_archives_rejected(name, kind):
    member = tarfile.TarInfo(name)
    member.type = {'file': tarfile.REGTYPE, 'symlink': tarfile.SYMTYPE,
                   'hardlink': tarfile.LNKTYPE, 'fifo': tarfile.FIFOTYPE}[kind]
    member.linkname = '../../escape'
    assert post(client(), archive(special=member)).status_code == 400


@pytest.mark.parametrize('change', ['missing_pickle', 'unsafe_pickle', 'custom_loader',
                                  'missing_contract', 'wrong_disease', 'unsafe_artifact',
                                  'source_manifest', 'missing_environment'])
def test_invalid_artifacts_rejected(change):
    contents = files()
    descriptor = yaml.safe_load(contents['MLmodel'])
    flavor = descriptor['flavors']['python_function']
    if change == 'missing_pickle':
        del contents['python_model.pkl']
    elif change == 'unsafe_pickle':
        flavor['python_model'] = '../outside.pkl'
    elif change == 'custom_loader':
        flavor['loader_module'] = 'uploaded_code'
    elif change == 'missing_contract':
        del flavor['artifacts']['license']
    elif change == 'wrong_disease':
        contents['artifacts/dataset_meta.json'] = '{"use_case":"SCD"}'
    elif change == 'unsafe_artifact':
        flavor['artifacts']['dataset_meta']['path'] = '../../outside.json'
    elif change == 'missing_environment':
        del contents['python_env.yaml']
    contents['MLmodel'] = yaml.safe_dump(descriptor)
    if change == 'source_manifest':
        del contents['MLmodel']
        contents['msi_manifest.json'] = '{"schema_version":1}'
    assert post(client(), archive(contents)).status_code == 400


def test_invalid_uploads_and_limits(monkeypatch):
    assert post(client(), b'not an archive').status_code == 400
    assert post(client(), archive(), use_case='other').status_code == 422
    monkeypatch.setattr(models, 'MAX_BUNDLE_BYTES', 1)
    assert post(client(), archive()).status_code == 413
    monkeypatch.setattr(models, 'MAX_BUNDLE_BYTES', 10000)
    monkeypatch.setattr(model_upload, 'MAX_EXTRACTED_BYTES', 1)
    assert post(client(), archive()).status_code == 413


def test_registration_failure_cleans_up(monkeypatch):
    saved = []
    def fail(folder, *args):
        saved.append(folder)
        raise RuntimeError('backend details must not leak')
    monkeypatch.setattr(models, 'register_bundle', fail)
    response = post(client(), archive())
    assert response.status_code == 502
    assert 'backend details' not in response.text
    assert not saved[0].exists()


def test_real_pyfunc_upload_preserves_pickle_and_only_loads_on_worker(tmp_path, monkeypatch):
    import mlflow
    folder = tmp_path / 'model'
    metadata = tmp_path / 'dataset_meta.json'
    metadata.write_text(json.dumps({'use_case': 'AML', 'variables': [{'name': 'Age'}]}))
    license_path = tmp_path / 'LICENSE.txt'
    license_path.write_text('MIT')
    marker = tmp_path / 'unpickled'

    class FederatedPyfunc(mlflow.pyfunc.PythonModel):
        def __init__(self):
            self.probe_state = True

        def predict(self, context, model_input, params=None):
            return model_input

        def create_local_learner(self, dataset_meta=None):
            return 'learner'

        def create_aggregator(self):
            return 'aggregator'

        def __setstate__(self, state):
            marker.write_text('unpickled')
            self.__dict__.update(state)

    mlflow.pyfunc.save_model(str(folder), python_model=FederatedPyfunc(), pip_requirements=[],
                            artifacts={'dataset_meta': str(metadata), 'license': str(license_path)})
    assert not marker.exists()
    original_files = {p.relative_to(folder).as_posix(): p.read_bytes()
                      for p in folder.rglob('*') if p.is_file()}
    tracking = (tmp_path / 'tracking').as_uri()
    monkeypatch.setattr(model_upload.config, 'MLFLOW_BASE_URL', tracking)
    monkeypatch.setattr(model_upload.config, 'MLFLOW_TRACKING_USERNAME', '')
    result = post(client(), archive(original_files, prefix='model/'))
    assert result.status_code == 201, result.text
    assert not marker.exists(), 'The API unpickled the uploaded model'
    response = result.json()
    assert response['model_id'] == yaml.safe_load(original_files['MLmodel'])['model_uuid']
    tracking_client = mlflow.tracking.MlflowClient(tracking_uri=tracking, registry_uri=tracking)
    downloaded = Path(tracking_client.download_artifacts(response['run_id'], 'model'))
    assert {p.relative_to(downloaded).as_posix(): p.read_bytes()
            for p in downloaded.rglob('*') if p.is_file()} == original_files
    assert tracking_client.get_run(response['run_id']).info.status == 'FINISHED'
    loaded = mlflow.pyfunc.load_model(str(downloaded))
    assert marker.exists()
    envelope = loaded.unwrap_python_model()
    assert envelope.create_local_learner() == 'learner'
    assert envelope.create_aggregator() == 'aggregator'
    # FL's existing metadata loader understands the preserved artifacts descriptor.
    from interfaces import mlflow_client
    monkeypatch.setattr(mlflow_client._initialized, 'value', True)
    monkeypatch.setattr(mlflow_client._configured, 'value', True)
    monkeypatch.setattr(mlflow_client._current_config, 'model_name', 'demo')
    monkeypatch.setattr(mlflow_client._current_config, 'model_version', response['model_version'])
    old_tracking, old_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    try:
        mlflow.set_tracking_uri(tracking)
        mlflow.set_registry_uri(tracking)
        assert mlflow_client.load_dataset_meta(loaded)['use_case'] == 'AML'
    finally:
        mlflow.set_tracking_uri(old_tracking)
        mlflow.set_registry_uri(old_registry)
    # A second upload adds a version under the same registered name.
    result = post(client(), archive(original_files))
    assert result.status_code == 201, result.text
    assert result.json()['model_version'] == '2'


def test_backend_failure_marks_run_failed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import mlflow.tracking
    backend = Mock()
    backend.get_experiment_by_name.return_value = SimpleNamespace(
        experiment_id='1', lifecycle_stage='active')
    backend.create_run.return_value = SimpleNamespace(info=SimpleNamespace(
        run_id='run', artifact_uri='file:///model-artifacts'))
    backend.log_artifacts.side_effect = RuntimeError('storage failed')
    monkeypatch.setattr(mlflow.tracking, 'MlflowClient', Mock(return_value=backend))
    with pytest.raises(RuntimeError, match='storage failed'):
        model_upload.register_bundle(tmp_path, 'experiment', 'AML', False,
                                     'demo', 'AVG', 'user', {})
    backend.set_terminated.assert_called_once_with('run', status='FAILED')
    backend.create_model_version.assert_not_called()


def test_deleted_experiment_rejected_before_artifact_upload(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fastapi import HTTPException
    import mlflow.tracking
    backend = Mock()
    backend.get_experiment_by_name.return_value = SimpleNamespace(
        experiment_id='1', lifecycle_stage='deleted')
    monkeypatch.setattr(mlflow.tracking, 'MlflowClient', Mock(return_value=backend))
    with pytest.raises(HTTPException) as error:
        model_upload.register_bundle(tmp_path, 'experiment', 'AML', False,
                                     'demo', 'AVG', 'user', {})
    assert error.value.status_code == 409
    backend.create_run.assert_not_called()
