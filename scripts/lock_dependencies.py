"""Create the workspace lock once, then check it without silently changing it."""
import argparse
import os
from pathlib import Path
import subprocess

from package_indexes import package_indexes

ROOT = Path(__file__).resolve().parents[1]


def lock_dependencies(root, env, *, update=False, sync=False, profile="none"):
    indexes = package_indexes(env)
    command = [env.get('UV_EXECUTABLE', 'uv'), 'lock', '--python', '3.11']
    if (root / 'uv.lock').exists() and not update:
        command.append('--check')
    index_args = ['--index', f"synthema={indexes['SYNTHEMA_PYPI_URL']}"]
    command += index_args
    process_env = dict(env)
    username = env.get('SYNTHEMA_PYPI_USERNAME', env.get('UV_INDEX_SYNTHEMA_USERNAME', ''))
    password = env.get('SYNTHEMA_PYPI_PASSWORD', env.get('UV_INDEX_SYNTHEMA_PASSWORD', ''))
    if bool(username) != bool(password):
        raise ValueError('Supply both PyPI username and password, or neither')
    if username:
        process_env['UV_INDEX_SYNTHEMA_USERNAME'] = username
        process_env['UV_INDEX_SYNTHEMA_PASSWORD'] = password
    subprocess.run(command, cwd=root, env=process_env, check=True)
    if sync:
        command = [env.get('UV_EXECUTABLE', 'uv'), 'sync', '--locked', '--all-packages', '--group', 'dev'] + index_args
        if profile != 'none':
            command += ['--extra', profile]
        subprocess.run(command, cwd=root, env=process_env, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update', action='store_true', help='Allow uv to update an existing lock after dependency changes')
    parser.add_argument('--sync', action='store_true', help='Also sync the workspace environment using the same indexes and credentials')
    parser.add_argument('--profile', choices=['none', 'aml-full', 'scd-full'], default='none')
    args = parser.parse_args()
    lock_dependencies(ROOT, os.environ, update=args.update, sync=args.sync, profile=args.profile)
