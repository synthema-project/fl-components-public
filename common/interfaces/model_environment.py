"""Provision compatible model requirements before MLflow unpickles user code.

The Flower process owns live framework objects. Resolve against its installed
versions and install only new distributions, never replace a live dependency.
"""
import fcntl
import importlib
from importlib import metadata
import logging
import os
from pathlib import Path
import re
import shutil
import site
import sysconfig
import subprocess
import sys
import tempfile
import tomllib
from urllib.parse import urlsplit

from packaging.requirements import Requirement, InvalidRequirement
from packaging.utils import canonicalize_name
import yaml

logger = logging.getLogger(__name__)
DEFAULT_INDEX = 'https://pypi.synthema.rid-intrasoft.eu/simple/'


def _requirements(folder):
    """MLflow's exported requirements honor explicit MSI overrides."""
    requirements = folder / 'requirements.txt'
    if requirements.is_file():
        return requirements.read_text()
    descriptor = yaml.safe_load((folder / 'MLmodel').read_text()) or {}
    project = descriptor.get('flavors', {}).get('python_function', {}).get('artifacts', {}).get('pyproject')
    if project:
        path = (folder / project['path']).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError('Model pyproject must be inside the model artifact')
    else:
        path = folder / 'pyproject.toml'
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        project = tomllib.load(stream).get('project', {})
    if 'dependencies' in project.get('dynamic', []):
        raise ValueError('Model requires exported static requirements; dynamic pyproject dependencies are unsupported')
    return '\n'.join(project.get('dependencies', []))


def _parse_requirements(text):
    requirements = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Index options and recursive files must come from deployment configuration.
        try:
            requirements.append(Requirement(re.split(r'\s+#', line, maxsplit=1)[0]))
        except InvalidRequirement:
            raise ValueError('Model requirements must contain PEP 508 dependencies, not pip options or nested files') from None
    return requirements


def _installed_versions():
    return {canonicalize_name(dist.metadata['Name']): dist.version
            for dist in metadata.distributions() if dist.metadata.get('Name')}


def _already_satisfied(requirements, installed):
    """Avoid network access when the complete requested graph is already present."""
    pending = [(req, {''}) for req in requirements]
    visited = set()
    while pending:
        req, parent_extras = pending.pop()
        if req.marker and not any(req.marker.evaluate({'extra': extra}) for extra in parent_extras):
            continue
        name = canonicalize_name(req.name)
        version = installed.get(name)
        if req.url or version is None or not req.specifier.contains(version, prereleases=True):
            return False
        key = (name, frozenset(req.extras))
        if key in visited:
            continue
        visited.add(key)
        try:
            children = metadata.requires(name) or []
            pending.extend((Requirement(child), {'', *req.extras}) for child in children)
        except (metadata.PackageNotFoundError, InvalidRequirement):
            return False
    return True


def _index_arguments():
    url = os.getenv('SYNTHEMA_PYPI_URL', DEFAULT_INDEX).strip()
    if not url:
        return []
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or any(c.isspace() for c in url)
            or not parsed.path.rstrip('/').endswith('/simple')):
        raise ValueError('SYNTHEMA_PYPI_URL must be an HTTPS /simple/ URL without credentials')
    return ['--index', 'synthema=' + url]


def _run_uv(command, env):
    try:
        subprocess.run(command, env=env, capture_output=True, text=True, check=True,
                       timeout=int(os.getenv('MODEL_DEPENDENCIES_TIMEOUT_SECONDS', '600')))
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('Model dependency provisioning timed out') from error
    except subprocess.CalledProcessError as error:
        # Resolver diagnostics are useful, but never expose index credentials/URLs.
        detail = re.sub(r'(?:https?|git\+https?)://\S+', '<package-source>', error.stderr or '')
        for key in ('UV_INDEX_SYNTHEMA_USERNAME', 'UV_INDEX_SYNTHEMA_PASSWORD',
                    'SYNTHEMA_PYPI_USERNAME', 'SYNTHEMA_PYPI_PASSWORD'):
            if env.get(key):
                detail = detail.replace(env[key], '<redacted>')
        raise RuntimeError('Cannot provision model dependencies without changing installed FL packages. '
                           'Check model requirements, index access and runtime versions. ' + detail[-4000:]) from error


def install_model_dependencies(folder):
    folder = Path(folder)
    if os.getenv('MODEL_DEPENDENCIES_ENABLED', 'true').lower() != 'true':
        logger.info('Model dependency provisioning disabled')
        return
    text = _requirements(folder)
    requirements = _parse_requirements(text)
    if not requirements:
        logger.info('Model has no declared runtime requirements')
        return
    env = dict(os.environ)
    env.setdefault('UV_CACHE_DIR', '/tmp/uv-model-cache')
    env['UV_NO_PROGRESS'] = '1'
    for suffix in ('USERNAME', 'PASSWORD'):
        if env.get('SYNTHEMA_PYPI_' + suffix):
            env['UV_INDEX_SYNTHEMA_' + suffix] = env['SYNTHEMA_PYPI_' + suffix]
    indexes = _index_arguments()
    # Lock the interpreter environment across threads/processes sharing this pod.
    import hashlib
    lock_name = hashlib.sha256(sys.executable.encode()).hexdigest()[:16]
    with open('/tmp/fl-model-environment-' + lock_name + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        installed = _installed_versions()
        for req in requirements:
            version = installed.get(canonicalize_name(req.name))
            applies = req.marker is None or req.marker.evaluate()
            if applies and version is not None and req.url is None and version not in req.specifier:
                raise RuntimeError(f'Model requires {req.name}{req.specifier}, but the running FL environment '
                                   f'contains {req.name}=={version}. Use compatible model requirements or rebuild the runtime.')
        if _already_satisfied(requirements, installed):
            logger.info('Model runtime dependencies already satisfied')
            return
        uv = shutil.which('uv')
        if uv is None:
            raise RuntimeError('uv is required to provision model dependencies; rebuild the FL runtime image')
        with tempfile.TemporaryDirectory(prefix='fl-model-requirements-') as directory:
            work = Path(directory)
            source, constraints, resolved = [work / name for name in ('model.txt', 'runtime.txt', 'resolved.txt')]
            source.write_text('\n'.join(str(req) for req in requirements) + '\n')
            constraints.write_text('\n'.join(f'{name}=={version}' for name, version in sorted(installed.items())) + '\n')
            logger.info('Resolving model dependencies against the running FL environment')
            _run_uv([uv, 'pip', 'compile', str(source), '--python', sys.executable,
                     '--constraint', str(constraints), '--output-file', str(resolved),
                     '--no-header', '--no-annotate', *indexes], env)
            new = [req for req in _parse_requirements(resolved.read_text())
                   if canonicalize_name(req.name) not in installed]
            if new:
                source.write_text('\n'.join(str(req) for req in new) + '\n')
                logger.info('Installing %d missing model dependency distributions with uv', len(new))
                _run_uv([uv, 'pip', 'install', '--python', sys.executable, '--no-deps',
                         '--requirement', str(source), '--constraint', str(constraints), *indexes], env)
                for path in {sysconfig.get_path("purelib"), sysconfig.get_path("platlib")}:
                    site.addsitedir(path)
                importlib.invalidate_caches()
            logger.info('Model runtime dependencies ready')


def load_model_with_dependencies(model_uri, loader):
    import mlflow.artifacts
    # Downloading artifacts does not deserialize the model. Install first.
    folder = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    install_model_dependencies(folder)
    # MLflow prepends the bundle's code directory to sys.path, but Python
    # otherwise reuses modules imported by a preceding model version.
    code = Path(folder) / 'code'
    if code.is_dir():
        roots = {path.stem for path in code.glob('*.py')}
        roots.update(path.name for path in code.iterdir() if path.is_dir())
        for name in list(sys.modules):
            if name.split('.')[0] in roots:
                del sys.modules[name]
        importlib.invalidate_caches()
    logger.info('Loading model artifact %s', model_uri)
    return loader(folder)
