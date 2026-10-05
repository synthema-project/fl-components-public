"""Keep training/pool-only deployments independent of reporting metadata."""
import os
from fl_client.metadata_contract import evaluation_settings


def utility_reporting_enabled(metadata):
    if os.getenv('VALIDATION_PROFILE') == 'none':
        return False
    return metadata is not None and bool(evaluation_settings(metadata).get('aml_utility'))


def full_reporting_enabled():
    return os.getenv('VALIDATION_PROFILE') in {'aml-full', 'scd-full'}
