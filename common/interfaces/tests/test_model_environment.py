"""Provisioning precedes unpickling and cannot replace live FL distributions."""
from pathlib import Path
from unittest.mock import Mock
import subprocess
import pytest
from interfaces import model_environment as runtime


def test_exported_requirements_override_manifest(tmp_path):
    (tmp_path / 'requirements.txt').write_text('numpy>=1\n')
    (tmp_path / 'pyproject.toml').write_text('[project]\ndependencies=["other"]\n')
    assert runtime._requirements(tmp_path) == 'numpy>=1\n'


def test_reads_bundled_manifest_without_optional_extras(tmp_path):
    (tmp_path / 'MLmodel').write_text('flavors:\n  python_function:\n    artifacts:\n      pyproject:\n        path: artifacts/pyproject.toml\n')
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'artifacts/pyproject.toml').write_text(
        '[project]\ndependencies=["numpy>=1"]\n[project.optional-dependencies]\ndev=["pytest"]\n')
    assert runtime._requirements(tmp_path) == 'numpy>=1'
    (tmp_path / 'MLmodel').write_text('flavors:\n  python_function:\n    artifacts:\n      pyproject:\n        path: ../outside.toml\n')
    with pytest.raises(ValueError, match='inside'):
        runtime._requirements(tmp_path)


@pytest.mark.parametrize('value', ['--index-url https://user:secret@example.test/simple', '-r other.txt', '-e .'])
def test_model_cannot_supply_installer_options(value):
    with pytest.raises(ValueError, match='PEP 508') as error:
        runtime._parse_requirements(value)
    assert 'secret' not in str(error.value)


def test_installs_only_new_distributions_with_frozen_runtime_constraints(tmp_path, monkeypatch):
    (tmp_path / 'requirements.txt').write_text('base>=1\nnew-dep==2\n')
    monkeypatch.setattr(runtime, '_installed_versions', lambda: {'base': '1.0'})
    monkeypatch.setattr(runtime.shutil, 'which', lambda _: '/bin/uv')
    monkeypatch.setenv('SYNTHEMA_PYPI_USERNAME', 'fixture-user')
    monkeypatch.setenv('SYNTHEMA_PYPI_PASSWORD', 'fixture-secret')
    calls = []
    def run(command, env):
        calls.append(command)
        assert 'fixture-secret' not in ' '.join(command)
        assert env['UV_INDEX_SYNTHEMA_PASSWORD'] == 'fixture-secret'
        pins = Path(command[command.index('--constraint') + 1]).read_text()
        assert pins == 'base==1.0\n'
        if 'compile' in command:
            Path(command[command.index('--output-file') + 1]).write_text('base==1.0\nnew-dep==2.0\n')
        else:
            assert '--no-deps' in command
            assert Path(command[command.index('--requirement') + 1]).read_text() == 'new-dep==2.0\n'
    monkeypatch.setattr(runtime, '_run_uv', run)
    runtime.install_model_dependencies(tmp_path)
    assert len(calls) == 2


def test_conflicting_model_fails_without_mutating_environment(tmp_path, monkeypatch):
    (tmp_path / 'requirements.txt').write_text('numpy==2\n')
    monkeypatch.setattr(runtime, '_installed_versions', lambda: {'numpy': '1.26.4'})
    monkeypatch.setattr(runtime.shutil, 'which', lambda _: '/bin/uv')
    run = Mock()
    monkeypatch.setattr(runtime, '_run_uv', run)
    with pytest.raises(RuntimeError, match='numpy==1.26.4'):
        runtime.install_model_dependencies(tmp_path)
    run.assert_not_called()


def test_load_order_and_failure_prevent_deserialization(monkeypatch):
    events = []
    monkeypatch.setattr('mlflow.artifacts.download_artifacts', lambda **kwargs: '/tmp/model')
    monkeypatch.setattr(runtime, 'install_model_dependencies', lambda path: events.append('installed'))
    def load(path):
        assert path == '/tmp/model'
        events.append('loaded')
    runtime.load_model_with_dependencies('models:/demo/1', load)
    assert events == ['installed', 'loaded']
    monkeypatch.setattr(runtime, 'install_model_dependencies', Mock(side_effect=RuntimeError('conflict')))
    loader = Mock()
    with pytest.raises(RuntimeError, match='conflict'):
        runtime.load_model_with_dependencies('models:/demo/1', loader)
    loader.assert_not_called()


def test_dependency_errors_redact_credentials(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr='Failed https://user:secret@example.test/simple secret')
    monkeypatch.setattr(runtime.subprocess, 'run', fail)
    with pytest.raises(RuntimeError) as error:
        runtime._run_uv(['uv'], {'UV_INDEX_SYNTHEMA_PASSWORD': 'secret'})
    assert 'secret' not in str(error.value)
    assert 'example.test' not in str(error.value)


def test_real_uv_installs_local_model_dependency_in_disposable_environment(tmp_path):
    """Exercise actual uv resolution/install without network or changing this venv."""
    import importlib.util
    import os
    import shutil
    import site
    import sys
    import zipfile
    uv = shutil.which('uv')
    if not uv:
        pytest.skip('uv executable unavailable')
    package = 'fl_model_dependency_probe'
    assert importlib.util.find_spec(package) is None
    wheel = tmp_path / (package + '-1.0-py3-none-any.whl')
    info = package + '-1.0.dist-info'
    members = {
        package + '/__init__.py': 'VALUE = "installed-by-uv"\n',
        info + '/METADATA': 'Metadata-Version: 2.1\nName: fl-model-dependency-probe\nVersion: 1.0\n',
        info + '/WHEEL': 'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
    }
    members[info + '/RECORD'] = ''.join(name + ',,\n' for name in [*members, info + '/RECORD'])
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    environment = tmp_path / 'venv'
    env = dict(os.environ, UV_OFFLINE='1', UV_NO_MANAGED_PYTHON='1',
               UV_CACHE_DIR=str(tmp_path / 'uv-cache'), UV_FIND_LINKS=str(tmp_path),
               SYNTHEMA_PYPI_URL='', MODEL_DEPENDENCIES_ENABLED='true')
    common = Path(runtime.__file__).resolve().parents[1]
    env['PYTHONPATH'] = os.pathsep.join([str(common), *site.getsitepackages()])
    subprocess.run([uv, 'venv', '--python', sys.executable, str(environment)],
                   env=env, check=True, capture_output=True, text=True)
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'requirements.txt').write_text('fl-model-dependency-probe==1.0\n')
    code = ('from interfaces.model_environment import install_model_dependencies; '
            f'install_model_dependencies({str(model)!r}); '
            'from fl_model_dependency_probe import VALUE; assert VALUE == "installed-by-uv"')
    completed = subprocess.run([str(environment / 'bin/python'), '-c', code],
                               env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert importlib.util.find_spec(package) is None


def test_satisfied_graph_needs_no_index_or_install(tmp_path, monkeypatch):
    (tmp_path / 'requirements.txt').write_text('base[extra]>=1\n')
    monkeypatch.setattr(runtime, '_installed_versions', lambda: {'base': '1.0', 'child': '2.0'})
    monkeypatch.setattr(runtime.metadata, 'requires', lambda name: {
        'base': ['child>=2; extra == "extra"'], 'child': []}[name])
    run = Mock()
    monkeypatch.setattr(runtime, '_run_uv', run)
    runtime.install_model_dependencies(tmp_path)
    run.assert_not_called()
    assert not runtime._already_satisfied(runtime._parse_requirements('base[extra]>=1'), {'base': '1.0'})


def test_successive_model_versions_load_their_own_bundled_code(tmp_path, monkeypatch):
    """Real MLflow deserialization must not reuse a prior bundle's module."""
    import importlib
    import sys
    import mlflow.pyfunc
    from mlflow.pyfunc import PythonModel

    class ProbeModel(PythonModel):
        def __init__(self, factory):
            self.factory = factory

        def predict(self, context, model_input, params=None):
            return self.factory()

    module_name = 'model_version_cache_probe'
    folders = []
    original_path = sys.path[:]
    try:
        for version in (1, 2):
            source = tmp_path / f'source{version}'
            source.mkdir()
            file = source / f'{module_name}.py'
            file.write_text(f'def value():\n    return {version}\n')
            sys.modules.pop(module_name, None)
            sys.path.insert(0, str(source))
            module = importlib.import_module(module_name)
            folder = tmp_path / f'model{version}'
            mlflow.pyfunc.save_model(str(folder), python_model=ProbeModel(module.value),
                                     code_paths=[str(file)], pip_requirements=[])
            folders.append(folder)
            sys.path.remove(str(source))
        sys.modules.pop(module_name, None)
        monkeypatch.setattr(runtime, 'install_model_dependencies', lambda _: None)
        monkeypatch.setattr('mlflow.artifacts.download_artifacts',
                            lambda artifact_uri: str(folders[int(artifact_uri) - 1]))
        # Demonstrate the original failure before exercising the runtime fix.
        assert mlflow.pyfunc.load_model(str(folders[0])).predict(None) == 1
        assert mlflow.pyfunc.load_model(str(folders[1])).predict(None) == 1
        for version in (1, 2, 1):
            model = runtime.load_model_with_dependencies(str(version), mlflow.pyfunc.load_model)
            assert model.predict(None) == version
    finally:
        sys.modules.pop(module_name, None)
        sys.path[:] = original_path
