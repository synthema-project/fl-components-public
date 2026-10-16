"""Validate and register existing MLflow pyfunc artifacts without loading their pickle."""
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import uuid

import yaml

from fastapi import HTTPException
from restapi import config

logger = logging.getLogger(__name__)
MAX_BUNDLE_BYTES = int(os.getenv('MODEL_UPLOAD_MAX_BYTES', str(50 * 1024 * 1024)))
MAX_EXTRACTED_BYTES = int(os.getenv('MODEL_UPLOAD_MAX_EXTRACTED_BYTES', str(200 * 1024 * 1024)))
MAX_FILES = 5000
_LOCK = threading.Lock()


def unpack_bundle(source, destination):
    """Extract only ordinary files/directories, with bounded size and safe paths."""
    total = 0
    seen = set()
    try:
        with tarfile.open(source, 'r:gz') as archive:
            for count, member in enumerate(archive, 1):
                path = PurePosixPath(member.name)
                if count > MAX_FILES:
                    raise ValueError("Bundle contains too many archive members")
                if member.isdir() and member.name.rstrip('/') in ('.', './'):
                    continue
                if (path.is_absolute() or '..' in path.parts
                        or '\\' in member.name or not path.parts
                        or not (member.isfile() or member.isdir())):
                    raise ValueError('Bundle contains an unsafe or unsupported archive member')
                normalized = str(path)
                if normalized in seen:
                    raise ValueError('Bundle contains duplicate paths')
                seen.add(normalized)
                total += member.size
                if total > MAX_EXTRACTED_BYTES:
                    raise HTTPException(413, 'Expanded model bundle is too large')
                target = destination.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
    except HTTPException:
        raise
    except (tarfile.TarError, OSError, ValueError, EOFError) as error:
        raise HTTPException(400, 'Invalid or unsafe model bundle') from error


def _local_path(folder, value, directory=False):
    """Require model descriptors to reference existing files inside the artifact."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError('Model artifact paths must be nonempty relative paths')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Model artifact paths must stay within the uploaded model')
    target = folder.joinpath(*path.parts)
    if not (target.is_dir() if directory else target.is_file()):
        raise ValueError('A file referenced by MLmodel is missing from the upload')
    return target


def validate_bundle(folder, use_case):
    """Inspect YAML/JSON descriptors only; never deserialize python_model.pkl."""
    try:
        # Support tar -C model . as well as an archive containing one model/ folder.
        if not (folder / 'MLmodel').is_file():
            children = list(folder.iterdir())
            if len(children) != 1 or not children[0].is_dir():
                raise ValueError('Upload must contain a complete MLflow model directory')
            folder = children[0]
        descriptor_path = folder / 'MLmodel'
        if descriptor_path.stat().st_size > 1024 * 1024:
            raise ValueError('MLmodel descriptor is too large')
        descriptor = yaml.safe_load(descriptor_path.read_text())
        if not isinstance(descriptor, dict):
            raise ValueError('MLmodel must be a YAML mapping')
        flavors = descriptor.get('flavors', {})
        if not isinstance(flavors, dict):
            raise ValueError('MLmodel flavors must be a mapping')
        flavor = flavors.get('python_function')
        if not isinstance(flavor, dict) or flavor.get('loader_module') != 'mlflow.pyfunc.model':
            raise ValueError('Upload must be an MLflow pickle-based python_model artifact')
        pickle_file = _local_path(folder, flavor.get('python_model'))
        if not pickle_file.stat().st_size:
            raise ValueError('The serialized Python model is empty')
        if flavor.get('code') is not None:
            _local_path(folder, flavor['code'], directory=True)
        env = flavor.get('env')
        if not isinstance(env, dict):
            raise ValueError('The model must include its MLflow environment descriptors')
        for key in ('conda', 'virtualenv'):
            _local_path(folder, env.get(key))
        _local_path(folder, 'requirements.txt')
        artifacts = flavor.get('artifacts')
        if not isinstance(artifacts, dict):
            raise ValueError('The model must include its dataset_meta and license artifacts')
        for artifact in artifacts.values():
            if not isinstance(artifact, dict):
                raise ValueError('Invalid MLflow artifact descriptor')
            # MLflow loads the local path; original uri values are provenance only.
            value = artifact.get('path')
            target = folder.joinpath(*PurePosixPath(value).parts) if isinstance(value, str) else None
            _local_path(folder, value, directory=bool(target and target.is_dir()))
        for name in ('dataset_meta', 'license'):
            if name not in artifacts:
                raise ValueError('The model must include its dataset_meta and license artifacts')
        metadata = json.loads(_local_path(folder, artifacts['dataset_meta'].get('path')).read_text())
        license_text = _local_path(folder, artifacts['license'].get('path')).read_text()
        if not isinstance(metadata, dict) or metadata.get('use_case') != use_case:
            raise ValueError('Metadata use_case does not match the requested use case')
        if not metadata.get('variables') or not license_text.strip():
            raise ValueError('Dataset variables and a nonempty license are required')
        # Optional evaluation contract can be bundled as an MLflow artifact or a root file.
        if 'evaluation_config' in artifacts:
            evaluation_path = _local_path(folder, artifacts['evaluation_config'].get('path'))
        else:
            evaluation_path = folder / 'evaluation_config.json'
        if evaluation_path.exists() and not isinstance(json.loads(evaluation_path.read_text()), dict):
            raise ValueError('Evaluation configuration must be a JSON object')
        return folder, descriptor
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        raise HTTPException(400, f'Invalid MLflow model artifact: {error}') from error


@contextmanager
def _artifact_tracking_uri(uri):
    """MLflow's proxied artifact repository also reads the global tracking URI."""
    import mlflow
    previous = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(uri)
        yield
    finally:
        mlflow.set_tracking_uri(previous)


def register_bundle(folder, experiment_name, use_case, trained, model_name,
                    aggregator_name, user_id, descriptor):
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    # SDK artifact upload does not import the model or install its requirements.
    # Retain existing service credentials, never a user's Keycloak token.
    with _LOCK, _artifact_tracking_uri(config.MLFLOW_BASE_URL):
        if config.MLFLOW_TRACKING_USERNAME:
            os.environ['MLFLOW_TRACKING_USERNAME'] = config.MLFLOW_TRACKING_USERNAME
            os.environ['MLFLOW_TRACKING_PASSWORD'] = config.MLFLOW_TRACKING_PASSWORD
        client = MlflowClient(tracking_uri=config.MLFLOW_BASE_URL,
                              registry_uri=config.MLFLOW_BASE_URL)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is not None and experiment.lifecycle_stage == 'deleted':
            raise HTTPException(409, 'The experiment is deleted; restore it before uploading')
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(experiment_name)
        run = client.create_run(experiment_id=experiment_id, tags={
            'use_case': use_case, 'trained': str(trained), 'uploaded_by': user_id,
            'model_name': model_name, 'aggregator_name': aggregator_name,
        })
        try:
            # Preserve MLmodel, pickle, code/, artifacts/ and environment files byte-for-byte.
            client.log_artifacts(run.info.run_id, str(folder), artifact_path='model')
            try:
                client.create_registered_model(model_name)
            except MlflowException as error:
                if error.error_code != 'RESOURCE_ALREADY_EXISTS':
                    raise
            source = run.info.artifact_uri.rstrip('/') + '/model'
            version = client.create_model_version(model_name, source=source, run_id=run.info.run_id)
            client.set_terminated(run.info.run_id, status='FINISHED')
            return {'message': 'Model uploaded and registered',
                    'model_id': descriptor.get('model_uuid') or str(uuid.uuid4()),
                    'run_id': run.info.run_id, 'model_uri': f'models:/{model_name}/{version.version}',
                    'experiment_id': str(experiment_id), 'model_version': str(version.version)}
        except Exception:
            try:
                client.set_terminated(run.info.run_id, status='FAILED')
            except Exception:
                logger.exception('Could not mark the failed model upload run')
            raise
