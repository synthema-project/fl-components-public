"""Federated context survives transport without changing MLflow step indexing."""
import pytest
from flwr.common import ParametersRecord
from interfaces.recordset import create_train_model_recordset, read_train_model_recordset


@pytest.mark.parametrize('iteration', [0, 4])
def test_round_context_round_trip(iteration):
    record = create_train_model_recordset(ParametersRecord(), iteration, 5)
    assert record.configs_records['config']['current_round'] == iteration + 1
    _, step, maximum = read_train_model_recordset(record)
    assert (step, maximum) == (iteration, 5)


@pytest.mark.parametrize('key,value', [('current_round', 0), ('current_round', 3),
                                      ('max_round', 0), ('max_round', True)])
def test_invalid_wire_context_rejected(key, value):
    record = create_train_model_recordset(ParametersRecord(), 0, 5)
    record.configs_records['config'][key] = value
    with pytest.raises(ValueError, match='round context'):
        read_train_model_recordset(record)
