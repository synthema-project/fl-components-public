"""Dataset partitions and feature metadata used by the MSI learner contract."""
from pathlib import Path

import pandas as pd
from model_store_interface import DatasetPartition
from fl_client.metadata_contract import parse_metadata, variables
import json


def partition_paths(paths):
    if isinstance(paths, dict):
        unknown = set(paths) - {'train', 'val', 'test', 'schema'}
        if unknown or 'train' not in paths:
            raise ValueError('Dataset role mapping must contain train and only train/val/test/schema roles')
        return paths
    roles = {}
    plain = []
    for path in paths:
        name = str(path).rsplit('/', 1)[-1].lower()
        role = next((r for r in ('train', 'val', 'test') if name.endswith(f'_{r}.csv')), None)
        if name.endswith('_meta.json'):
            role = 'schema'
        if role:
            if role in roles:
                raise ValueError(f'Multiple {role} objects: select one dataset collection in the catalogue')
            roles[role] = path
        elif name.endswith('.csv'):
            plain.append(path)
        else:
            raise ValueError(f'Unsupported dataset object: {name}')
    if roles:
        if plain or 'train' not in roles:
            raise ValueError('Mixed full datasets and partitions, or missing train partition; select one dataset collection')
        return roles
    if not plain:
        raise ValueError('No training CSV objects supplied')
    return {'train': plain}


def read_frame(path):
    frame = pd.read_csv(path)
    if len(frame.columns) == 1 and any(s in str(frame.columns[0]) for s in (',', ';', '\t')):
        raise ValueError('Stored CSV contains a whole delimited row in one column; re-upload the original with the fixed ingestor')
    frame = frame.loc[:, ~frame.columns.str.match(r'^Unnamed')]
    if frame.empty:
        raise ValueError('Dataset partition is empty')
    return frame


def load_partitions(paths, download, evaluation=None):
    roles = partition_paths(paths)
    def local(path):
        return path if str(path).startswith(('/', '.')) else download(path)
    partitions = {}
    for role in ('train', 'val', 'test'):
        if role in roles:
            files = roles[role] if isinstance(roles[role], list) else [roles[role]]
            frames = [read_frame(local(p)) for p in files]
            if any(list(f.columns) != list(frames[0].columns) for f in frames[1:]):
                raise ValueError('Training CSV files have inconsistent columns')
            partitions[role] = pd.concat(frames, ignore_index=True)
    schema = parse_metadata(json.loads(Path(local(roles['schema'])).read_text()), evaluation) if 'schema' in roles else None
    return partitions, schema


def prepare_partitions(learner, partitions, metadata):
    if metadata is None:
        raise ValueError('MSI learner requires a validated DatasetMeta feature schema from the dataset or registered model artifact')
    required = [variable.name for variable in variables(metadata)]
    prepared = {}
    for role in ('train', 'val', 'test'):
        if role not in partitions:
            continue
        data = partitions[role]
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f'{role} partition missing schema columns: {", ".join(sorted(missing))}')
        result = learner.prepare_data(data=data.copy(), partition=DatasetPartition(role), dataset_meta=metadata)
        if result is None:
            raise ValueError(f'MSI prepare_data returned no data for {role}')
        prepared[role] = result
    return prepared


def ensure_utility_holdout(partitions, config):
    """Keep explicit test partitions, or apply an explicitly configured split."""
    if "test" in partitions:
        return partitions
    fraction = config.get("holdout_fraction")
    if fraction is None:
        raise ValueError("AML utility needs a test partition; upload partitioned data or explicitly configure holdout_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(partitions["train"], test_size=fraction,
                                  random_state=int(config.get("random_state", 42)))
    if len(train) < 10 or len(test) < 10:
        raise ValueError("AML split requires at least 10 training and 10 holdout rows")
    return {**partitions, "train": train.copy(), "test": test.copy()}
