"""MSI-compatible authenticated model registration."""
import logging
from pathlib import Path
import tempfile

from fastapi import Depends, File, Form, HTTPException, UploadFile
from restapi.auth import UserClaims, require_authentication
from restapi.helpers.model_upload import (
    MAX_BUNDLE_BYTES, unpack_bundle, validate_bundle, register_bundle,
)

logger = logging.getLogger(__name__)


def upload_model(
    bundle: UploadFile = File(...),
    experiment_name: str = Form(..., min_length=1, max_length=255),
    use_case: str = Form(...),
    trained: bool = Form(False),
    model_name: str = Form(..., min_length=1, max_length=255),
    aggregator_name: str = Form(..., min_length=1, max_length=255),
    user: UserClaims = Depends(require_authentication),
):
    if use_case not in ('AML', 'SCD'):
        raise HTTPException(422, 'Unsupported use case')
    if any(not value.strip() or any(ord(c) < 32 for c in value)
           for value in (experiment_name, model_name, aggregator_name)):
        raise HTTPException(422, 'Names must be nonempty and contain no control characters')
    with tempfile.TemporaryDirectory(prefix='msi-upload-') as temporary:
        root = Path(temporary)
        archive = root / 'bundle.tar.gz'
        size = 0
        with archive.open('wb') as output:
            while chunk := bundle.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BUNDLE_BYTES:
                    raise HTTPException(413, 'Model bundle is too large')
                output.write(chunk)
        folder = root / 'model_bundle'
        folder.mkdir()
        unpack_bundle(archive, folder)
        folder, descriptor = validate_bundle(folder, use_case)
        try:
            return register_bundle(folder, experiment_name, use_case, trained, model_name,
                                   aggregator_name, user.sub, descriptor)
        except HTTPException:
            raise
        except Exception:
            logger.exception('MLflow model registration failed')
            raise HTTPException(502, 'MLflow model registration failed')
