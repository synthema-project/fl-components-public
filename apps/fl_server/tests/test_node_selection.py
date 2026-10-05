from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from flwr.common import ConfigsRecord, RecordSet
from fl_server.requires import resolve_node_selection, load_data
from fl_server.schemas.task import TaskRead


def reply(node_id, name, participate=True):
    return SimpleNamespace(
        has_error=lambda: False,
        metadata=SimpleNamespace(src_node_id=node_id),
        content=RecordSet(configs_records={"config": ConfigsRecord({
            "participate": participate, "node_name": name,
        })}),
    )


def test_selection_uses_names_and_excludes_other_clients():
    assert resolve_node_selection([reply(891, 'NODE1'), reply(432, 'NODE2')], ['NODE2']) == {432: 'NODE2'}


def test_omitted_selection_includes_eligible_clients():
    assert resolve_node_selection([reply(891, 'NODE1'), reply(432, 'NODE2', False)]) == {891: 'NODE1'}


@pytest.mark.parametrize('replies,selection,match', [
    ([reply(891, 'NODE1')], ['NODE2'], 'unavailable'),
    ([reply(891, 'NODE1', False)], ['NODE1'], 'unavailable'),
    ([reply(891, 'NODE1'), reply(432, 'NODE1')], ['NODE1'], 'Multiple'),
    ([reply(891, '')], ['NODE1'], 'NODE_NAME'),
    ([], None, 'No eligible'),
])
def test_invalid_selection_fails_before_training(replies, selection, match):
    with pytest.raises(ValueError, match=match):
        resolve_node_selection(replies, selection)


def test_task_queue_payload_keeps_selection():
    task = TaskRead.model_validate_json('''{"id":1,"user_id":"test","use_case":"AML",
        "model_name":"model","model_version":1,"num_global_iterations":1,
        "run_name":"test","experiment_name":"test","selected_nodes":["NODE2"]}''')
    assert task.selected_nodes == ['NODE2']


def test_dataset_assignment_uses_reported_identity():
    with patch('fl_server.requires.DataCatalogueClient') as catalogue, \
         patch('fl_server.requires.execution_flow') as flow, \
         patch('fl_server.requires.interfaces.recordset.create_load_data_recordset') as record:
        catalogue.return_value.get_use_case_datasets.return_value = {
            'NODE1': ['unselected.csv'], 'NODE2': ['selected.csv']}
        load_data(Mock(), [432], 'AML', ('{}', '{}'), node_names={432: 'NODE2'})
        flow.call_args.kwargs['per_node_recordset_factory'](432)
        record.assert_called_once_with('AML', ['selected.csv'], '{}', '{}')


def test_missing_datasets_prevent_dispatch():
    with patch('fl_server.requires.DataCatalogueClient') as catalogue, \
         patch('fl_server.requires.execution_flow') as flow:
        catalogue.return_value.get_use_case_datasets.return_value = {'NODE1': ['data.csv']}
        with pytest.raises(ValueError, match='No catalogue datasets.*NODE2'):
            load_data(Mock(), [432], 'AML', ('{}', '{}'), node_names={432: 'NODE2'})
        flow.assert_not_called()
