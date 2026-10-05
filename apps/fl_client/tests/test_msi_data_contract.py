"""Current MSI documents and partitions, using only artificial test data."""
import json
from unittest.mock import patch
import pandas as pd
import pytest
from testing import configuration, create_partitions, dataset_model, Learner, model
from fl_client import stages
from fl_client.data_contract import prepare_partitions, read_frame
from interfaces import mlflow_client, recordset


def test_preparation_training_upload_preserves_fitted_parameters(tmp_path, monkeypatch):
    monkeypatch.setenv('VALIDATION_PROFILE', 'aml-full')
    state = {'node_name': 'ICH'}
    metadata, evaluation = configuration()
    stages.load_data('AML', create_partitions(tmp_path), state, json.dumps(metadata), json.dumps(evaluation))
    with patch.object(mlflow_client, 'load_model', return_value=model()), patch.object(mlflow_client, 'load_dataset_meta', return_value=None):
        stages.load_model(state)
    stages.prepare_data(state)
    assert len(state['data']) == 72
    assert len(state['prepared_partitions']['test']) == 24
    with patch.object(mlflow_client, 'log_metrics'):
        assert stages.train_model(state, 1)['n_samples'] == 72
    parameters = stages.get_model_parameters(state)
    with patch.object(mlflow_client, 'upload_final_state', return_value={'model_id': 'trained'}) as upload:
        stages.upload_model(parameters, state)
    assert upload.call_args.kwargs == {'use_case': 'AML'}
    assert (state['local_learner'].mean != 0).any()


def test_flower_nested_paths_are_json_encoded(tmp_path):
    paths = create_partitions(tmp_path)
    metadata, evaluation = configuration()
    record = recordset.create_load_data_recordset('AML', paths, json.dumps(metadata), json.dumps(evaluation))
    assert recordset.read_load_data_recordset(record)[1] == paths


def test_whole_row_csv_is_rejected(tmp_path):
    path = tmp_path / 'bad.csv'; path.write_text('"Age,HB"\n"40,10"\n')
    with pytest.raises(ValueError, match='single|one column'):
        read_frame(path)


def test_missing_schema_and_missing_columns_fail_before_training():
    frame = pd.DataFrame({'HB': [10]})
    with pytest.raises(ValueError, match='schema'):
        prepare_partitions(Learner(), {'train': frame}, None)
    with pytest.raises(ValueError, match='missing schema columns'):
        prepare_partitions(Learner(), {'train': frame}, dataset_model())


def test_full_reporting_requires_an_independent_holdout(monkeypatch):
    monkeypatch.setenv('VALIDATION_PROFILE', 'aml-full')
    with pytest.raises(ValueError, match='independent'):
        stages.prepare_data({'local_learner': Learner(), 'model_meta': dataset_model(),
                             'data_partitions': {'train': pd.DataFrame({'Age': [40]})}})
