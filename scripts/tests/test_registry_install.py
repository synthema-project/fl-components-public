"""Exercise the Docker install shell without downloading private packages."""
import os
from pathlib import Path
import subprocess

import pytest

APPS = Path(__file__).resolve().parents[2] / 'apps'


def install(app, profile):
    dockerfile = (APPS / app / 'Dockerfile').read_text()
    command = dockerfile.split('    set -- --index ', 1)[1].split('\n\n', 1)[0]
    command = 'set -- --index ' + command.replace('\\\n', '')
    # Replace uv with a shell function to capture the arguments of the actual RUN command.
    return subprocess.run(
        ['sh', '-ec', 'uv() { printf "%s\\n" "$@"; }; ' + command],
        env={**os.environ, 'VALIDATION_PROFILE': profile,
             'SYNTHEMA_PYPI_URL': 'https://packages.example/simple/'},
        capture_output=True, text=True,
    )


@pytest.mark.parametrize('app,profile,extra', [
    ('fl_server', 'none', None), ('fl_client', 'none', None),
    ('restapi', 'none', None), ('status_listener', 'none', None),
    ('fl_client', 'aml-full', 'aml-full'), ('fl_client', 'scd-full', 'scd-full'),
])
def test_registry_install_selects_indexes_and_profile(app, profile, extra):
    result = install(app, profile)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[0] == 'sync'
    assert '--locked' in args
    assert '--no-install-workspace' in args
    names = {'fl_client': 'synthema-fl-client', 'fl_server': 'synthema-fl-server',
             'restapi': 'synthema-fl-restapi', 'status_listener': 'status-listener'}
    assert args[args.index('--package') + 1] == names[app]
    pos = args.index('synthema=https://packages.example/simple/')
    assert args[pos - 1] == '--index'
    assert args.count('--index') == 1
    if extra:
        assert args[args.index('--extra') + 1] == extra
    else:
        assert '--extra' not in args


def test_unsupported_profile_fails_before_install():
    result = install('fl_client', 'unknown')
    assert result.returncode != 0
    assert not result.stdout
