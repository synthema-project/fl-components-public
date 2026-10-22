from types import SimpleNamespace
import pandas as pd
from flwr.common import ConfigsRecord
from fl_client.custom_metadata import compute, apply


def test_extracts_only_raw_training_data_before_preparation():
    seen = []
    def extract(data, metadata):
        seen.append((data.id.tolist(), metadata))
        data.iloc[0, 0] = -1
        return ConfigsRecord({'categories': b'bytes-preserved', 'max_id': 2})
    learner = SimpleNamespace(compute_custom_metadata=extract, set_custom_metadata=seen.append)
    state = {'local_learner': learner, 'model_meta': 'schema',
             'data_partitions': {'train': pd.DataFrame({'id': [1, 2]}),
                                 'test': pd.DataFrame({'id': [999]})}}
    result = compute(state)
    assert seen == [([1, 2], 'schema')]
    assert state['data_partitions']['train'].id.tolist() == [1, 2]
    apply(result, state)
    assert seen[-1]['categories'] == b'bytes-preserved'


def test_missing_hooks_are_optional():
    state = {'local_learner': object()}
    assert compute(state) is None
    apply(ConfigsRecord({'max_id': 2}), state)
