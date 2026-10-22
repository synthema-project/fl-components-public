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
from restapi.routers import models, mlflow_router


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
    app.include_router(mlflow_router.router)
    if authenticated:
        app.dependency_overrides[require_authentication] = get_mock_user
    return TestClient(app)


def post(c, bundle, **changes):
    return c.post('/mlflow/models/upload',
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


def test_mlflow_router_exposes_authenticated_upload(monkeypatch):
    from restapi.routers.mlflow_router import router
    app = FastAPI()
    app.include_router(router)
    monkeypatch.setattr('restapi.auth.AUTH_DISABLE_TESTS', False)
    with TestClient(app) as api:
        assert api.post('/mlflow/models/upload', data={'experiment_name': 'test', 'use_case': 'AML',
            'model_name': 'demo', 'aggregator_name': 'AVG'},
            files={'bundle': ('model.tar.gz', archive(), 'application/gzip')}).status_code == 401
    app.dependency_overrides[require_authentication] = get_mock_user
    register = Mock(return_value={'model_uri': 'models:/demo/1', 'model_version': '1'})
    monkeypatch.setattr(models, 'register_bundle', register)
    with TestClient(app) as api:
        response = api.post('/mlflow/models/upload', data={'experiment_name': 'test', 'use_case': 'AML',
            'model_name': 'demo', 'aggregator_name': 'AVG'},
            files={'bundle': ('model.tar.gz', archive(), 'application/gzip')})
    assert response.status_code == 201, response.text
    register.assert_called_once()


def test_submission_bundle_round_trip_through_mlflow_router(tmp_path, monkeypatch):
    """Exercise the MSI-compatible bundle contract without an example model/repo."""
    import mlflow
    from restapi import config
    module = tmp_path / 'upload_contract_model.py'
    module.write_text("""
class Learner:
    def train_round(self):
        return {"loss": 0.0}
class Aggregator:
    def aggregate_parameters(self, parameters):
        return parameters
""")
    monkeypatch.syspath_prepend(str(tmp_path))

    class SubmissionEnvelope(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return model_input

        def create_local_learner(self, dataset_meta=None):
            from upload_contract_model import Learner
            return Learner()

        def create_aggregator(self):
            from upload_contract_model import Aggregator
            return Aggregator()

    artifacts = {}
    for name, filename, content in [
        ('dataset_meta', 'dataset_meta.json', json.dumps({
            'use_case': 'AML', 'variables': [{'name': 'age', 'datatype': 'Text'}]})),
        ('evaluation_config', 'evaluation_config.json', json.dumps({
            'utility_evaluation': {'clinical_variables': ['age']},
            'privacy_evaluation': {'known_attributes': ['age'], 'sensitive_attributes': []}})),
        ('license', 'LICENSE.txt', 'MIT'),
    ]:
        artifact = tmp_path / filename
        artifact.write_text(content)
        artifacts[name] = str(artifact)
    folder = tmp_path / 'submission'
    mlflow.pyfunc.save_model(str(folder), python_model=SubmissionEnvelope(),
                            artifacts=artifacts, code_paths=[str(module)], pip_requirements=[])
    tracking = (tmp_path / 'registry').as_uri()
    monkeypatch.setattr(config, 'MLFLOW_BASE_URL', tracking)
    monkeypatch.setattr(config, 'MLFLOW_TRACKING_USERNAME', '')
    monkeypatch.setattr(config, 'MLFLOW_TRACKING_PASSWORD', '')
    bundle = archive({p.relative_to(folder).as_posix(): p.read_bytes()
                      for p in folder.rglob('*') if p.is_file()})
    response = post(client(), bundle)
    assert response.status_code == 201, response.text
    result = response.json()
    assert result['model_version'] == '1'
    registry = mlflow.tracking.MlflowClient(tracking_uri=tracking, registry_uri=tracking)
    saved = Path(registry.download_artifacts(result['run_id'], 'model'))
    # Loading happens on the worker side, after the API has preserved the bundle.
    loaded = mlflow.pyfunc.load_model(str(saved)).unwrap_python_model()
    assert loaded.create_local_learner().train_round() == {'loss': 0.0}
    assert loaded.create_aggregator().aggregate_parameters([1, 2]) == [1, 2]
    assert (saved / 'code' / module.name).is_file()
    for artifact in artifacts.values():
        assert (saved / 'artifacts' / Path(artifact).name).read_bytes() == Path(artifact).read_bytes()


@pytest.mark.parametrize('fail', [False, True])
def test_proxy_artifacts_use_http_tracking_and_restore_previous_uri(tmp_path, monkeypatch, fail):
    from types import SimpleNamespace
    import mlflow
    from mlflow.store.artifact.mlflow_artifacts_repo import MlflowArtifactsRepository
    tracking = 'http://mlflow:5000'
    previous = mlflow.get_tracking_uri()
    backend = Mock()
    backend.get_experiment_by_name.return_value = SimpleNamespace(experiment_id='1', lifecycle_stage='active')
    backend.create_run.return_value = SimpleNamespace(info=SimpleNamespace(
        run_id='run', artifact_uri='mlflow-artifacts:/1/run/artifacts'))
    backend.create_model_version.return_value = SimpleNamespace(version='1')
    def upload(*args, **kwargs):
        assert mlflow.get_tracking_uri() == tracking
        resolved = MlflowArtifactsRepository.resolve_uri(
            'mlflow-artifacts:/1/run/artifacts', mlflow.get_tracking_uri())
        assert resolved.startswith(tracking + '/api/2.0/mlflow-artifacts/artifacts/')
        if fail:
            raise RuntimeError('artifact upload failed')
    backend.log_artifacts.side_effect = upload
    monkeypatch.setattr(mlflow.tracking, 'MlflowClient', Mock(return_value=backend))
    monkeypatch.setattr(model_upload.config, 'MLFLOW_BASE_URL', tracking)
    monkeypatch.setattr(model_upload.config, 'MLFLOW_TRACKING_USERNAME', '')
    try:
        mlflow.set_tracking_uri((tmp_path / 'mlruns').as_uri())
        local_default = mlflow.get_tracking_uri()
        if fail:
            with pytest.raises(RuntimeError, match='artifact upload failed'):
                model_upload.register_bundle(tmp_path, 'test', 'AML', False, 'demo', 'AVG', 'user', {})
        else:
            result = model_upload.register_bundle(tmp_path, 'test', 'AML', False, 'demo', 'AVG', 'user', {})
            assert result['model_version'] == '1'
        assert mlflow.get_tracking_uri() == local_default
    finally:
        mlflow.set_tracking_uri(previous)
