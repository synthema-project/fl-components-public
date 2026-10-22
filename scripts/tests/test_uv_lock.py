"""Lock lifecycle and authentication must be consistent with image installs."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]


@pytest.fixture
def locks(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('lock_dependencies_test', SCRIPTS / 'lock_dependencies.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_then_check_and_intentional_update(locks, tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(locks.subprocess, 'run', run)
    locks.lock_dependencies(tmp_path, {})
    assert '--check' not in run.call_args.args[0]
    (tmp_path / 'uv.lock').write_text('fixture')
    locks.lock_dependencies(tmp_path, {})
    assert '--check' in run.call_args.args[0]
    locks.lock_dependencies(tmp_path, {}, update=True)
    assert '--check' not in run.call_args.args[0]
    assert all(call.kwargs['check'] for call in run.call_args_list)


def test_sync_preserves_indexes_and_uses_env_credentials(locks, tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(locks.subprocess, 'run', run)
    env = {'SYNTHEMA_PYPI_URL': 'https://packages.example/simple/',
           'SYNTHEMA_PYPI_USERNAME': 'private-user', 'SYNTHEMA_PYPI_PASSWORD': 'private-password'}
    locks.lock_dependencies(tmp_path, env, sync=True, profile='scd-full')
    assert run.call_count == 2
    lock, sync = [call.args[0] for call in run.call_args_list]
    assert sync[1] == 'sync'
    assert '--locked' in sync and '--all-packages' in sync
    assert sync[sync.index('--extra') + 1] == 'scd-full'
    index = 'synthema=https://packages.example/simple/'
    assert index in lock and index in sync
    for call in run.call_args_list:
        assert call.args[0].count('--index') == 1
        assert call.kwargs['env']['UV_INDEX_SYNTHEMA_PASSWORD'] == 'private-password'
        assert 'private-password' not in ' '.join(call.args[0])
        assert 'private-user' not in ' '.join(call.args[0])


def test_stale_lock_stops_before_sync(locks, tmp_path, monkeypatch):
    (tmp_path / 'uv.lock').write_text('fixture')
    run = Mock(side_effect=locks.subprocess.CalledProcessError(1, 'uv lock --check'))
    monkeypatch.setattr(locks.subprocess, 'run', run)
    with pytest.raises(locks.subprocess.CalledProcessError):
        locks.lock_dependencies(tmp_path, {}, sync=True)
    assert run.call_count == 1


@pytest.mark.parametrize('env', [
    {'SYNTHEMA_PYPI_PASSWORD': 'password-only'},
    {'SYNTHEMA_PYPI_URL': 'https://user:password@host/simple/'},
])
def test_bad_inputs_fail_before_running_uv(locks, tmp_path, monkeypatch, env):
    run = Mock()
    monkeypatch.setattr(locks.subprocess, 'run', run)
    with pytest.raises(ValueError):
        locks.lock_dependencies(tmp_path, env)
    run.assert_not_called()
