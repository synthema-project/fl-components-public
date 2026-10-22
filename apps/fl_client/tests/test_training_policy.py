"""Restore original task-controlled routes; MSI DP stays inside the model."""
from unittest.mock import patch
import importlib
import pytest


@pytest.mark.parametrize('mode,smpc,route', [
    (None, False, 'train_model_if'),
    ('local_dp', False, 'train_model_if'),
    ('central_dp', False, 'train_model_fixed_clipping_if'),
    (None, True, 'train_model_SMPC_if'),
    ('local_dp', True, 'train_model_SMPC_if'),
])
def test_original_training_dispatch(monkeypatch, mode, smpc, route):
    with patch('interfaces.mlflow_client.setup_mlflow'):
        main = importlib.import_module('fl_client.main')
    monkeypatch.setattr(main, 'global_vars', {'dp_config': {'dp_mode': mode}, 'SMPC_enabled': smpc})
    with patch.object(main, 'train_model_if', return_value='normal') as normal, patch.object(
        main, 'train_model_fixed_clipping_if', return_value='central',
    ) as central, patch.object(main, 'train_model_SMPC_if', return_value='smpc') as secure:
        expected = dict(train_model_if=normal, train_model_fixed_clipping_if=central, train_model_SMPC_if=secure)
        assert main.train('request', None) == expected[route].return_value
        expected[route].assert_called_once_with('request', main.global_vars)
        assert sum(mock.call_count for mock in expected.values()) == 1


def test_original_central_dp_and_smpc_rejection(monkeypatch):
    with patch('interfaces.mlflow_client.setup_mlflow'):
        main = importlib.import_module('fl_client.main')
    monkeypatch.setattr(main, 'global_vars', {'dp_config': {'dp_mode': 'central_dp'}, 'SMPC_enabled': True})
    with pytest.raises(RuntimeError, match='Cannot use both SMPC and DP'):
        main.train('request', None)


def test_central_handler_uses_original_clipping_with_round_context():
    from types import SimpleNamespace
    from collections import OrderedDict
    import numpy as np
    from fl_models.iris.utils import Utils
    from interfaces.recordset import create_train_model_recordset
    from fl_client.interface import train_model_fixed_clipping_if
    from flwr.common import MetricsRecord, array_from_numpy, ParametersRecord
    baseline = ParametersRecord({'weights': array_from_numpy(np.array([0., 0.]))})
    updated = ParametersRecord({'weights': array_from_numpy(np.array([6., 8.]))})
    request = SimpleNamespace(content=create_train_model_recordset(baseline, 2, 5),
                              create_reply=lambda content, ttl: SimpleNamespace(content=content))
    state = {'dp_config': {'clipping_norm': 1}}
    with patch('fl_client.interface.stages.set_model_parameters'), patch(
        'fl_client.interface.stages.train_model', return_value=MetricsRecord({'loss': .2}),
    ) as train, patch('fl_client.interface.stages.get_model_parameters', return_value=updated):
        reply = train_model_fixed_clipping_if(request, state)
    train.assert_called_once_with(state, 2, 5)
    np.testing.assert_allclose(Utils.parameters_to_dict(reply.content.parameters_records['parameters'])['weights'], [.6, .8])


def test_disabled_task_clears_previous_platform_dp():
    from interfaces.recordset import create_set_dp_cfg_recordset, read_set_dp_cfg_recordset
    from fl_client import stages
    state = {'dp_config': {'dp_mode': 'central_dp', 'clipping_norm': 1}}
    stages.set_dp_config(read_set_dp_cfg_recordset(create_set_dp_cfg_recordset(None, None, None)), state)
    assert state['dp_config'] is None
