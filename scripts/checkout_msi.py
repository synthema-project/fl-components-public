"""Check out an MSI branch using an optional GitHub token from the environment."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit


def checkout(env):
    repository = env.get('MSI_REPOSITORY', '')
    try:
        url = urlsplit(repository)
        valid = (url.scheme == 'https' and url.hostname and url.username is None
                 and url.password is None and not url.query and not url.fragment
                 and not any(c.isspace() for c in repository))
        if env.get('MSI_GITHUB_TOKEN'):
            valid = valid and url.hostname == 'github.com' and url.port in (None, 443)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('MSI_REPOSITORY must be a credential-free HTTPS URL; token access requires github.com')

    branch = env.get('MSI_BRANCH', '').strip()
    subprocess.run(['git', 'check-ref-format', f'refs/heads/{branch}'],
                   env=env, check=True, stdout=sys.stderr)
    with tempfile.TemporaryDirectory(prefix='msi-askpass-') as folder:
        helper = Path(folder) / 'askpass'
        # No token is written to disk: Git invokes this script to read the environment.
        helper.write_text(f'#!{sys.executable}\n' + '''import os, sys
prompt = sys.argv[1]
if 'Username' in prompt:
    print('x-access-token')
elif 'Password' in prompt:
    print(os.environ.get('MSI_GITHUB_TOKEN', ''))
else:
    sys.exit(1)
''')
        helper.chmod(0o700)
        git_env = {**env, 'GIT_ASKPASS': str(helper), 'GIT_TERMINAL_PROMPT': '0'}
        commands = [
            ['git', 'init', '-q'],
            ['git', 'remote', 'add', 'origin', repository],
            ['git', '-c', 'credential.helper=', '-c', 'credential.username=x-access-token',
             'fetch', '--no-tags', 'origin',
             f'+refs/heads/{branch}:refs/remotes/origin/{branch}'],
            ['git', '-c', 'core.hooksPath=/dev/null', 'checkout', '--detach',
             f'refs/remotes/origin/{branch}'],
        ]
        for command in commands:
            subprocess.run(command, env=git_env, check=True, stdout=sys.stderr)
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       env=git_env, text=True).strip()


if __name__ == '__main__':
    try:
        print(checkout(dict(os.environ)))
    except ValueError as error:
        raise SystemExit(str(error))
    except subprocess.CalledProcessError:
        raise SystemExit('MSI checkout failed; see Git diagnostics above')
