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


@pytest.mark.parametrize('method', ['get_parameters', 'train_round'])
def test_invalid_learner_responses_are_rejected(state, method):
    with patch.object(state['local_learner'], method, return_value='invalid'), patch.object(mlflow_client, 'log_metrics'):
        with pytest.raises(TypeError):
            if method == 'get_parameters':
                stages.get_model_parameters(state)
            elif method == 'train_round':
                stages.train_model(state, 1, 5)
            else:
                stages.evaluate_model(state)


def test_round_evaluation_preserves_independent_test_set(state):
    holdout = state['prepared_partitions']['test'].copy(deep=True)
    with patch.object(state['local_learner'], 'evaluate') as evaluate:
        assert not stages.evaluate_model(state)
    evaluate.assert_not_called()
    pd.testing.assert_frame_equal(state['prepared_partitions']['test'], holdout)


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


@pytest.mark.parametrize('iteration', [0, 4])
def test_msi_receives_round_context_and_tracking_records_it(state, iteration):
    received = []

    def train_round(data, dataset_meta, current_round, max_round):
        received.append((current_round, max_round))
        return MetricsRecord({'loss': 0.5})

    state['local_learner'].train_round = train_round
    with patch.object(mlflow_client, 'log_metrics') as log:
        metrics = stages.train_model(state, iteration, 5)
    assert received == [(iteration + 1, 5)]
    assert dict(metrics) == {'loss': 0.5}
    log.assert_called_once_with(
        {'loss': 0.5, 'fl/current_round': iteration + 1, 'fl/max_round': 5}, iteration)


@pytest.mark.parametrize('iteration,maximum', [(-1, 5), (5, 5), (0, 0), (True, 5)])
def test_invalid_round_context_is_rejected(state, iteration, maximum):
    with pytest.raises(ValueError, match='round context'):
        stages.train_model(state, iteration, maximum)


@pytest.mark.parametrize('empty', [False, True])
def test_round_evaluation_skips_missing_or_empty_validation(state, empty):
    if empty:
        state['prepared_partitions']['val'] = pd.DataFrame()
    else:
        state['prepared_partitions'].pop('val', None)
    with patch.object(state['local_learner'], 'evaluate') as evaluate:
        assert not stages.evaluate_model(state, 1, 5)
    evaluate.assert_not_called()


def test_round_evaluation_ignores_historical_validation(state):
    state['prepared_partitions']['val'] = state['prepared_partitions']['test'].copy()
    with patch.object(state['local_learner'], 'evaluate') as evaluate:
        assert not stages.evaluate_model(state, 1, 5)
    evaluate.assert_not_called()
    with pytest.raises(ValueError, match='round context'):
        stages.evaluate_model(state, 5, 5)


def test_evaluation_message_applies_parameters_and_returns_metrics():
    from flwr.common import Message, Metadata, MessageType, ParametersRecord, DEFAULT_TTL
    from fl_client.interface import evaluate_model_if
    from interfaces.recordset import create_evaluate_model_recordset
    parameters = ParametersRecord()
    request = Message(
        content=create_evaluate_model_recordset(parameters, 1, 5),
        metadata=Metadata(run_id=1, message_id='eval', src_node_id=0,
                          dst_node_id=1, reply_to_message='', group_id='evaluate_model',
                          ttl=DEFAULT_TTL, message_type=MessageType.EVALUATE),
    )
    state = {}
    with patch.object(stages, 'set_model_parameters') as apply, patch.object(
        stages, 'evaluate_model', return_value=MetricsRecord({'loss': 0.25}),
    ) as evaluate:
        reply = evaluate_model_if(request, state)
    assert apply.call_args.args[0] is parameters
    evaluate.assert_called_once_with(state, 1, 5)
    assert reply.content.configs_records['config']['success']
    assert reply.content.metrics_records['metrics']['loss'] == 0.25


def test_step_zero_precedes_creation_and_uses_only_training_data(state):
    from interfaces.step_zero import aggregate
    from types import SimpleNamespace
    from testing import dataset_model
    state['data_partitions']['test']['Age'] = 1e10
    metadata = dataset_model()
    created_with = []
    def factory(dataset_meta):
        created_with.append(dataset_meta)
        return Learner()
    model = SimpleNamespace(unwrap_python_model=lambda: SimpleNamespace(create_local_learner=factory))
    with patch.object(stages.mlflow_client, 'load_model', return_value=model), patch.object(
        stages.mlflow_client, 'load_dataset_meta', return_value=None,
    ):
        payload = stages.compute_step_zero(state)
    assert state['local_learner'] is None
    assert not created_with
    enriched = aggregate([payload])
    stages.apply_step_zero(enriched, state)
    assert len(created_with) == 1
    age = next(v for v in created_with[0].metadata.variables if v.name == 'Age')
    assert age.mean == pytest.approx(state['data_partitions']['train'].Age.mean())
    stages.prepare_data(state)
    assert state['model_meta'] is created_with[0]


def test_step_zero_messages_exchange_summaries_before_learner_creation(state):
    from fl_client.interface import step_zero_if
    from interfaces.recordset import create_step_zero_recordset
    from flwr.common import Message, Metadata, MessageType, DEFAULT_TTL
    from interfaces.step_zero import aggregate
    def message(mode, metadata=None):
        return Message(content=create_step_zero_recordset(mode, metadata), metadata=Metadata(
            run_id=1, message_id=mode, src_node_id=0, dst_node_id=1,
            reply_to_message='', group_id='step_zero', ttl=DEFAULT_TTL,
            message_type=MessageType.QUERY))
    with patch.object(stages, 'load_model_metadata'):
        reply = step_zero_if(message('compute_step_zero'), state)
    payload = json.loads(reply.content.configs_records['step_zero']['payload_json'])
    assert set(payload) == {'metadata', 'metrics'}
    assert set(payload['metrics'][0]) == {'feature_name', 'count', 'sum_val', 'sum_sq_val'}
    assert state['local_learner'] is None
    with patch.object(stages, 'initialize_learner') as initialize:
        reply = step_zero_if(message('apply_step_zero', aggregate([payload])), state)
    assert reply.content.configs_records['config']['success']
    initialize.assert_called_once_with(state)


def test_model_dp_metrics_reach_aggregator_unchanged_but_mlflow_gets_scalars(state):
    from flwr.common import MetricsRecord
    metrics = MetricsRecord({'privacy_budget': [2.5, 1e-5], 'loss': 0.2})
    with patch.object(state['local_learner'], 'train_round', return_value=metrics), patch.object(
        stages.mlflow_client, 'log_metrics',
    ) as log:
        assert stages.train_model(state, 1, 5) is metrics
    assert metrics['privacy_budget'] == [2.5, 1e-5]
    logged = log.call_args.args[0]
    assert logged['privacy/epsilon'] == 2.5
    assert 'privacy_budget' not in logged
    assert stages.get_privacy_budget(state) == {'epsilon': 2.5, 'delta': 1e-5}


def test_privacy_budget_hook_takes_precedence_and_missing_is_not_zero(state):
    state['local_learner'].get_privacy_spent = lambda: {'epsilon': 3., 'delta': 1e-5}
    assert stages.get_privacy_budget(state)['epsilon'] == 3.
    del state['local_learner'].get_privacy_spent
    state.pop('reported_privacy_budget', None)
    assert stages.get_privacy_budget(state) is None


def test_actual_msi_dp_wrapper_runs_through_model_training_path(state):
    from model_store_interface.dp.dp_learner_wrapper import DPLocalLearnerWrapper
    class RoundLearner(Learner):
        def train_round(self, data, dataset_meta, current_round, max_round):
            assert (current_round, max_round) == (1, 2)
            return super().train_round(data, dataset_meta)
    wrapped = DPLocalLearnerWrapper(RoundLearner(), epsilon=1., delta=1e-5)
    state['local_learner'] = wrapped
    with patch.object(stages.mlflow_client, 'log_metrics'):
        metrics = stages.train_model(state, 0, 2)
    assert 'privacy_budget' in metrics
    with patch.object(wrapped._dp_wrapper, 'apply', wraps=wrapped._dp_wrapper.apply) as perturb:
        parameters = stages.get_model_parameters(state)
    perturb.assert_called_once()
    assert parameters
    reported = stages.get_privacy_budget(state)
    assert reported['epsilon'] > 0
    assert reported['delta'] == 1e-5
