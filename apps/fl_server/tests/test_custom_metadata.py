from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from flwr.common import ConfigsRecord
from fl_server.custom_metadata import exchange


def test_aggregates_all_supported_nodes_and_applies_global_metadata():
    received = []
    def aggregate(values):
        assert all(isinstance(v, ConfigsRecord) for v in values)
        return ConfigsRecord({'max_id': max(v['max_id'] for v in values), 'blob': b'unchanged'})
    model = SimpleNamespace(set_custom_metadata=received.append)
    state = {'aggregator': SimpleNamespace(aggregate_custom_metadata=aggregate),
             'model': SimpleNamespace(unwrap_python_model=lambda: model)}
    assert exchange(state, lambda: [ConfigsRecord({'max_id': 2}), None,
                                   ConfigsRecord({'max_id': 7})], received.append)
    assert len(received) == 2
    assert all(value['max_id'] == 7 and value['blob'] == b'unchanged' for value in received)


@pytest.mark.parametrize('aggregator,values', [
    (object(), [{'max_id': 2}]),
    (SimpleNamespace(aggregate_custom_metadata=lambda values: {}), [None]),
    (SimpleNamespace(aggregate_custom_metadata=lambda values: None), [{'max_id': 2}]),
])
def test_optional_hooks_or_empty_results_skip_broadcast(aggregator, values):
    broadcast = Mock()
    assert not exchange({'aggregator': aggregator}, lambda: values, broadcast)
    broadcast.assert_not_called()


def test_hook_errors_stop_execution():
    def aggregate(values):
        raise RuntimeError('model metadata failure')
    broadcast = Mock()
    with pytest.raises(RuntimeError, match='model metadata failure'):
        exchange({'aggregator': SimpleNamespace(aggregate_custom_metadata=aggregate)},
                 lambda: [ConfigsRecord({'max_id': 2})], broadcast)
    broadcast.assert_not_called()


def test_flower_exchange_uses_native_records_and_waits_for_acknowledgements(monkeypatch):
    from fl_server import requires
    from flwr.common import RecordSet
    events = []
    def flow(*args, **kwargs):
        request = kwargs['recordset_factory'](*kwargs['recordset_factory_args'])
        mode = request.configs_records['config']['mode']
        events.append(mode)
        if mode == 'compute_custom_metadata':
            return [SimpleNamespace(content=RecordSet(configs_records={
                'config': ConfigsRecord({'success': True}),
                'custom_metadata': ConfigsRecord({'max_id': value})})) for value in [3, 9]]
        assert request.configs_records['custom_metadata']['max_id'] == 9
        return [object(), object()]
    monkeypatch.setattr(requires, 'execution_flow', flow)
    state = {'aggregator': SimpleNamespace(aggregate_custom_metadata=lambda values:
                ConfigsRecord({'max_id': max(v['max_id'] for v in values)})),
             'model': SimpleNamespace(unwrap_python_model=lambda: object())}
    assert requires.exchange_custom_metadata(object(), [1, 2], state)
    assert events == ['compute_custom_metadata', 'set_custom_metadata']


def test_missing_client_reply_fails_instead_of_aggregating_subset(monkeypatch):
    from fl_server import requires
    monkeypatch.setattr(requires, 'execution_flow', lambda *a, **kw: [])
    aggregate = Mock()
    with pytest.raises(RuntimeError, match='every client reply'):
        requires.exchange_custom_metadata(object(), [1], {'aggregator': SimpleNamespace(
            aggregate_custom_metadata=aggregate)})
    aggregate.assert_not_called()
