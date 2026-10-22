"""Distributed summaries must reproduce MSI's local Step 0 result."""
import copy
import numpy as np
import pandas as pd
import pytest
from testing import dataset_model
from interfaces.step_zero import compute, aggregate
from model_store_interface.step_zero import execute_step_zero


def test_statistics_match_simulation_with_unequal_nodes_and_missing_values():
    metadata = dataset_model()
    frames = [pd.DataFrame({'Age': [10., np.nan, 30.], 'HB': [1., 2., 3.]}),
              pd.DataFrame({'Age': [40.], 'HB': [4.]})]
    payload = metadata.model_dump(mode='json')
    original = copy.deepcopy(payload)
    result = aggregate([{'metadata': payload, 'metrics': compute(frame, payload)} for frame in frames])
    reference = execute_step_zero(frames, metadata)
    for expected in reference.metadata.variables:
        actual = next(v for v in result['metadata']['variables'] if v['name'] == expected.name)
        assert actual.get('mean') == expected.mean
        assert actual.get('std') == expected.std
    assert payload == original
    assert pd.isna(frames[0].Age.iloc[1])
    age = next(v for v in result['metadata']['variables'] if v['name'] == 'Age')
    assert age['mean'] == 25.  # Sample-weighted, not an average of client means.


def test_legacy_models_skip_but_mixed_metadata_fails():
    empty = {'metadata': None, 'metrics': []}
    assert aggregate([empty, empty]) is None
    with pytest.raises(ValueError, match='every client'):
        aggregate([empty, {'metadata': dataset_model().model_dump(mode='json'), 'metrics': []}])


def test_incompatible_schemas_fail():
    metadata = dataset_model().model_dump(mode='json')
    other = copy.deepcopy(metadata)
    other['metadata']['variables'][0]['name'] = 'different'
    with pytest.raises(ValueError, match='incompatible'):
        aggregate([{'metadata': metadata, 'metrics': []}, {'metadata': other, 'metrics': []}])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.])
def test_invalid_statistics_fail(value):
    payload = dataset_model().model_dump(mode='json')
    with pytest.raises(ValueError):
        aggregate([{'metadata': payload, 'metrics': [dict(
            feature_name='Age', count=value, sum_val=1., sum_sq_val=1.)]}])
