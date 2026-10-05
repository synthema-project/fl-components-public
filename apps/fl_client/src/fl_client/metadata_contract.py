"""Bridge published MSI metadata versions without inventing evaluation settings."""
import json


def parse_metadata(payload, evaluation=None):
    if evaluation is not None:
        from model_store_interface.metadata.models.model import SynthemaDatasetModel
        return SynthemaDatasetModel.from_json(payload, evaluation)
    if 'metadata' in payload and 'evaluation_configuration' in payload:
        from model_store_interface.metadata.models.model import SynthemaDatasetModel
        return SynthemaDatasetModel.model_validate(payload)
    try:
        from model_store_interface import DatasetMeta
    except ImportError as error:
        raise ValueError('This MSI version requires both metadata and evaluation configuration') from error
    return DatasetMeta.model_validate(payload)


def variables(metadata):
    return metadata.metadata.variables if hasattr(metadata, 'metadata') else metadata.variables


def evaluation_settings(metadata):
    if metadata is None:
        return {}
    if hasattr(metadata, 'evaluation_configuration'):
        return metadata.evaluation_configuration.to_json()
    return metadata.evaluation or {}


def documents(metadata):
    if hasattr(metadata, 'to_json'):
        return metadata.to_json()
    return metadata.for_publish(), evaluation_settings(metadata)
