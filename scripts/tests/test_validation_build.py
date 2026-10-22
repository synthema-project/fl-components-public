import importlib.util
from pathlib import Path
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prepare_validation', ROOT/'scripts/prepare_validation.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def wheel(directory, name, adapter=False):
    path = directory / f'{name}-0.5.0-py3-none-any.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(f'{name}-0.5.0.dist-info/METADATA', f'Name: {name}\nVersion: 0.5.0\nProvides-Extra: aml-utility\n')
        if adapter: archive.writestr('model_store_interface/aml_utility.py', '')
    return path


def test_aml_rejects_stale_msi(tmp_path):
    wheel(tmp_path, 'model_store_interface')
    with pytest.raises(ValueError, match='lacks aml_utility'):
        prepare.prepare(tmp_path, tmp_path, 'aml')


def test_profile_and_wheel_staging(tmp_path):
    source=tmp_path/'sources'; source.mkdir()
    wheel(source,'model_store_interface',True); wheel(source,'sd_quality')
    for app in ['fl_client','fl_server']:(tmp_path/'apps'/app).mkdir(parents=True)
    prepare.prepare(tmp_path,source,'aml')
    assert len(list((tmp_path/'apps/fl_client/wheels').glob('*.whl'))) == 2
    assert len(list((tmp_path/'apps/fl_server/wheels').glob('*.whl'))) == 1
    with pytest.raises(ValueError,match='Unsupported'):
        prepare.prepare(tmp_path,source,'scd')


def test_image_test_failure_propagates(monkeypatch):
    import sys, types
    from unittest.mock import Mock
    docker=types.ModuleType('docker');docker.DockerClient=object
    errors=types.ModuleType('docker.errors')
    for name in ['ImageNotFound','APIError','BuildError']:setattr(errors,name,type(name,(Exception,),{}))
    monkeypatch.setitem(sys.modules,'docker',docker);monkeypatch.setitem(sys.modules,'docker.errors',errors)
    spec=importlib.util.spec_from_file_location('image_build_test',ROOT/'scripts/images.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    client=Mock();client.containers.run.side_effect=RuntimeError('pytest failed')
    with pytest.raises(RuntimeError,match='pytest failed'):
        module.test_images(client,{'images':[{'name':'client','test':True}]},'test')


@pytest.mark.parametrize('exit_code', [0, 1])
def test_image_tests_stream_stdout_and_clean_up(monkeypatch, capsys, exit_code):
    import sys, types
    from unittest.mock import Mock
    docker = types.ModuleType('docker'); docker.DockerClient = object
    errors = types.ModuleType('docker.errors')
    for name in ['ImageNotFound', 'APIError', 'BuildError']:
        setattr(errors, name, type(name, (Exception,), {}))
    monkeypatch.setitem(sys.modules, 'docker', docker)
    monkeypatch.setitem(sys.modules, 'docker.errors', errors)
    spec = importlib.util.spec_from_file_location('image_log_test', ROOT/'scripts/images.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    client = Mock()
    container = client.containers.run.return_value
    container.logs.return_value = [b'test_example FAILED\n', b'assert 1 == 2\n']
    container.wait.return_value = {'StatusCode': exit_code}
    if exit_code:
        with pytest.raises(RuntimeError, match='exit 1'):
            module.test_images(client, {'images': [{'name': 'client', 'test': True}]}, 'ci-test')
    else:
        module.test_images(client, {'images': [{'name': 'client', 'test': True}]}, 'ci-test')
    assert 'assert 1 == 2' in capsys.readouterr().out
    container.logs.assert_called_once_with(stream=True, follow=True, stdout=True, stderr=True)
    container.remove.assert_called_once_with(force=True)


@pytest.mark.parametrize('profile', ['aml-full', 'scd-full'])
def test_full_profile_requires_current_msi_contract_and_disease_extra(tmp_path, profile):
    wheel(tmp_path, 'model_store_interface')
    with pytest.raises(ValueError, match='disease extra'):
        prepare.prepare(tmp_path, tmp_path, profile)
