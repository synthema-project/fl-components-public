"""Artificial data and a small learner for offline integration tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from flwr.common import Array, array_from_numpy, ParametersRecord, MetricsRecord
from model_store_interface import DatasetPartition
from model_store_interface.metadata.models.model import SynthemaDatasetModel


def configuration():
    metadata = {'title': 'Artificial AML', 'description': 'Integration fixture, no patient data',
                'use_case': 'AML', 'version': '1', 'schema_version': '1',
                'theme': ['Health'], 'health_category': ['Electronic Health Records (EHR)',
                'Human genetic, epigenomic and genomic data', 'Data from medical registries and mortality registries'],
                'variables': [{'name': name, 'datatype': 'Number', 'description': name}
                              for name in ('Age', 'HB', 'PLT', 'OS_Status')]}
    evaluation = {'utility_evaluation': {'clinical_variables': ['Age', 'HB', 'PLT'], 'genomic_variables': []},
                  'privacy_evaluation': {'known_attributes': ['Age'], 'sensitive_attributes': ['OS_Status'],
                                         'differential_privacy': False}}
    return metadata, evaluation


def dataset_model():
    return SynthemaDatasetModel.from_json(*configuration())


def create_partitions(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({'Age': rng.normal(50, 5, 120), 'HB': rng.normal(10, 1, 120),
                          'PLT': rng.normal(150, 10, 120), 'OS_Status': rng.integers(0, 2, 120)})
    paths = {}
    for name, data in [('train', frame.iloc[:96]), ('test', frame.iloc[96:])]:
        path = folder / f'{name}.csv'; data.to_csv(path, index=False); paths[name] = str(path)
    path = folder / 'schema.json'; path.write_text(json.dumps(configuration()[0])); paths['schema'] = str(path)
    return paths


class Learner:
    def __init__(self):
        self.mean = np.zeros(4)
        self.prepared = {}

    def prepare_data(self, data, partition, dataset_meta):
        frame = data[[v.name for v in dataset_meta.metadata.variables]].astype(float)
        self.prepared[partition] = frame.copy()
        return frame

    def train_round(self, data, dataset_meta):
        self.mean = data.mean().to_numpy()
        return MetricsRecord({'n_samples': len(data)})

    def evaluate(self, data, dataset_meta):
        return MetricsRecord({'n_samples': len(data)})

    def get_parameters(self):
        return ParametersRecord({'mean': array_from_numpy(self.mean)})

    def set_parameters(self, parameters):
        self.mean = parameters['mean'].numpy().copy()

    def generate_synthetic_data(self, n_samples, dataset_meta):
        rng = np.random.default_rng(123)
        return pd.DataFrame(rng.normal(self.mean, 1, (n_samples, 4)),
                            columns=[v.name for v in dataset_meta.metadata.variables])


class Aggregator:
    def aggregate_parameters(self, parameters):
        return ParametersRecord({'mean': array_from_numpy(np.mean([p['mean'].numpy() for p in parameters], axis=0))})

    def aggregate_metrics(self, metrics):
        return MetricsRecord({'n_samples': sum(m['n_samples'] for m in metrics)})


def model():
    return SimpleNamespace(unwrap_python_model=lambda: SimpleNamespace(
        create_local_learner=lambda dataset_meta=None: Learner(), create_aggregator=Aggregator))
