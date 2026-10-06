import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

CHECKOUT = Path(__file__).resolve().parents[1] / 'checkout_msi.py'


@pytest.mark.parametrize('fail_fetch', [False, True])
def test_token_authentication_and_helper_cleanup(tmp_path, fail_fetch):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    git = bin_dir / 'git'
    git.write_text('''#!/usr/bin/python3
import json, os, pathlib, subprocess, sys
args = sys.argv[1:]
assert os.environ['MSI_GITHUB_TOKEN'] not in ' '.join(args)
if 'fetch' in args:
    helper = os.environ['GIT_ASKPASS']
    assert os.environ['MSI_GITHUB_TOKEN'] not in pathlib.Path(helper).read_text()
    user = subprocess.check_output([helper, 'Username for https://github.com:'], text=True).strip()
    password = subprocess.check_output([helper, 'Password for https://github.com:'], text=True).strip()
    assert user == 'x-access-token'
    assert password == os.environ['MSI_GITHUB_TOKEN']
    assert 'credential.helper=' in args
    pathlib.Path(os.environ['EVIDENCE']).write_text(json.dumps({'helper': helper, 'args': args}))
    if os.environ['FAIL_FETCH'] == '1':
        sys.exit(1)
if args == ['rev-parse', 'HEAD']:
    print('a' * 40)
''')
    git.chmod(0o700)
    evidence = tmp_path / 'evidence.json'
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}',
           'MSI_REPOSITORY': 'https://github.com/example/msi.git',
           'MSI_BRANCH': ' feature/msi ', 'MSI_GITHUB_TOKEN': 'fixture-private-token',
           'EVIDENCE': str(evidence), 'FAIL_FETCH': '1' if fail_fetch else '0'}
    result = subprocess.run([sys.executable, str(CHECKOUT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert (result.returncode != 0) == fail_fetch
    if not fail_fetch:
        assert result.stdout.strip() == 'a' * 40
    captured = json.loads(evidence.read_text())
    assert not Path(captured['helper']).exists()
    assert '+refs/heads/feature/msi:refs/remotes/origin/feature/msi' in captured['args']
    assert 'fixture-private-token' not in result.stdout + result.stderr


@pytest.mark.parametrize('repository', [
    'https://github.com.evil.example/msi.git',
    'https://user:password@github.com/example/msi.git',
    'http://github.com/example/msi.git',
    'https://github.com:8443/example/msi.git',
])
def test_rejects_token_to_unexpected_hosts(tmp_path, repository):
    env = {**os.environ, 'MSI_REPOSITORY': repository, 'MSI_BRANCH': 'main',
           'MSI_GITHUB_TOKEN': 'fixture-private-token'}
    result = subprocess.run([sys.executable, str(CHECKOUT)], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / '.git').exists()
    assert 'fixture-private-token' not in result.stdout + result.stderr
