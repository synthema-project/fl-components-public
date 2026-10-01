"""Stage errors, state handling and model publication with the current MSI contract."""
from unittest.mock import Mock, patch
import pandas as pd
import pytest
from flwr.common import MetricsRecord
from testing import Learner, dataset_model, model, create_partitions, configuration
from fl_client import stages
from interfaces import mlflow_client
import json


@pytest.fixture
def state(tmp_path):
    metadata, evaluation = configuration()
    result = {'node_name': 'ICH', 'local_learner': Learner()}
    stages.load_data('AML', create_partitions(tmp_path), result, json.dumps(metadata), json.dumps(evaluation))
    stages.prepare_data(result)
    return result


def test_load_data_missing_paths():
    with pytest.raises(ValueError, match='No dataset paths'):
        stages.load_data('AML', [], {'node_name': 'ICH'})


def test_load_model_uses_loaded_schema(state):
    with patch.object(mlflow_client, 'load_model', return_value=model()), patch.object(mlflow_client, 'load_dataset_meta', return_value=None):
        stages.load_model(state)
    assert isinstance(state['local_learner'], Learner)


@pytest.mark.parametrize('method', ['get_parameters', 'train_round', 'evaluate'])
def test_invalid_learner_responses_are_rejected(state, method):
    with patch.object(state['local_learner'], method, return_value='invalid'), patch.object(mlflow_client, 'log_metrics'):
        with pytest.raises(TypeError):
            if method == 'get_parameters':
                stages.get_model_parameters(state)
            elif method == 'train_round':
                stages.train_model(state, 1)
            else:
                stages.evaluate_model(state)


def test_evaluation_uses_validation_partition(state):
    with patch.object(state['local_learner'], 'evaluate', return_value=MetricsRecord({'n_samples': 24})) as evaluate:
        result = stages.evaluate_model(state)
    assert len(evaluate.call_args.args[0]) == 24
    assert result['n_samples'] == 24


def test_upload_keeps_final_parameters(state):
    state['local_learner'].mean[:] = 5
    parameters = stages.get_model_parameters(state)
    with patch.object(mlflow_client, 'upload_final_state', return_value={'model_id': 'trained'}) as upload:
        stages.upload_model(parameters, state)
    upload.assert_called_once_with(state['local_learner'], use_case='AML')
    assert (state['local_learner'].mean == 5).all()


def test_run_configuration_and_cleanup():
    with patch.object(stages, 'mlflow_client') as tracking:
        stages.set_run_config('experiment', 'parent', 'ICH', 'model', 1, 'train.csv')
        tracking.clean_current_config.assert_called_once_with('FAILED')
        tracking.create_child_run.assert_called_once_with('experiment', 'parent', 'ICH')
        tracking.set_dataset_signature.assert_called_once()
    with patch.object(mlflow_client, 'clean_current_config') as cleanup:
        stages.clean_current_config()
        cleanup.assert_called_once()
