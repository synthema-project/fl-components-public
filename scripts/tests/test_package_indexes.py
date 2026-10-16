import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('package_indexes', SCRIPTS / 'package_indexes.py')
indexes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indexes)


def test_shared_index_and_default():
    assert indexes.package_indexes({}) == {'SYNTHEMA_PYPI_URL': indexes.DEFAULT_INDEX}
    env = {'SYNTHEMA_PYPI_URL': 'https://packages.example/simple/'}
    assert indexes.package_indexes(env) == env


@pytest.mark.parametrize('url', ['', 'http://host/simple/', 'https://user:secret@host/simple/',
                                 'https://host/package.whl', 'https://host/simple/?token=secret',
                                 'https://host/simple/#secret', 'https://host:bad/simple/'])
def test_rejects_invalid_or_credential_bearing_urls(url):
    with pytest.raises(ValueError) as error:
        indexes.package_indexes({'SYNTHEMA_PYPI_URL': url})
    assert 'secret' not in str(error.value)


def test_build_passes_urls_and_secret_references(monkeypatch, tmp_path):
    docker = types.ModuleType('docker')
    docker.DockerClient = object
    errors = types.ModuleType('docker.errors')
    for name in ('ImageNotFound', 'APIError', 'BuildError'):
        setattr(errors, name, type(name, (Exception,), {}))
    monkeypatch.setitem(sys.modules, 'docker', docker)
    monkeypatch.setitem(sys.modules, 'docker.errors', errors)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('images_indexes_test', SCRIPTS / 'images.py')
    images = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(images)
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(images.subprocess, 'run', run)
    monkeypatch.setenv('SYNTHEMA_PYPI_URL', 'https://packages.example/simple/')
    monkeypatch.setenv('VALIDATION_PROFILE', 'aml-full')
    monkeypatch.setenv('UV_INDEX_SYNTHEMA_USERNAME', 'private-user')
    monkeypatch.setenv('UV_INDEX_SYNTHEMA_PASSWORD', 'private-password')
    assert images.build_image(None, {'name': 'synthema-fl-client', 'path': '.', 'msi': True}, 'test', tmp_path)
    cmd = run.call_args.args[0]
    assert cmd[-3:] == ['--file', str(tmp_path / 'Dockerfile'), str(tmp_path)]
    assert run.call_count == 1  # Only docker build; no local MSI checkout/build.
    assert 'SYNTHEMA_PYPI_URL=https://packages.example/simple/' in cmd
    assert sum(arg.startswith('SYNTHEMA_PYPI_URL=') for arg in cmd) == 1
    assert 'VALIDATION_PROFILE=aml-full' in cmd
    assert 'id=synthema_password,env=UV_INDEX_SYNTHEMA_PASSWORD' in cmd
    assert 'private-user' not in ' '.join(cmd)
    assert 'private-password' not in ' '.join(cmd)
