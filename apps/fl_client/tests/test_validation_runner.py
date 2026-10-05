"""Verify evaluator inputs, report artifacts and failures without patient data."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pandas as pd
import pytest
from fl_client import validation_runner as runner


def test_plugins_receive_training_holdout_and_shared_pool(tmp_path, monkeypatch):
    metadata = {'variables': [{'name': 'age'}, {'name': 'outcome'}]}
    (tmp_path / 'configuration.json').write_text(json.dumps({'metadata': metadata,
        'evaluation': {'privacy_evaluation': {'known_attributes': ['age']}}, 'use_case': 'AML'}))
    for name, values in [('real', [1, 2]), ('holdout', [3, 4]), ('synthetic', [5, 6])]:
        pd.DataFrame({'age': values, 'outcome': [0, 1]}).to_pickle(tmp_path / (name + '.pkl'))
    inputs = []
    def evaluator(**kwargs):
        inputs.append(kwargs)
        folder = Path(kwargs['output_folder'])
        (folder / 'chart.png').write_bytes(b'artificial-image')
        (folder / 'report.html').write_text('<html><img src="chart.png"></html>')
    monkeypatch.setattr(runner, 'validator', lambda kind, disease: evaluator)
    runner.evaluate(tmp_path)
    assert len(inputs) == 3
    assert inputs[0]['real_data'].age.tolist() == [1, 2]
    assert inputs[0]['holdout_data'].age.tolist() == [3, 4]
    assert inputs[0]['synth_data'].age.tolist() == [5, 6]
    assert inputs[1]['evaluation_config']['privacy_evaluation']['format'] == 'html'
    for kind in ('utility', 'privacy', 'fidelity'):
        assert 'data:image/png;base64,' in (tmp_path / (kind + '.html')).read_text()
    assert json.loads((tmp_path / 'results.json').read_text()) == ['utility', 'privacy', 'fidelity']


def test_missing_or_ambiguous_plugins_fail_without_fallback(monkeypatch):
    monkeypatch.setattr(runner.importlib.metadata, 'entry_points', lambda **kwargs: [])
    with pytest.raises(RuntimeError, match='No installed'):
        runner.validator('utility', 'AML')
    monkeypatch.setattr(runner.importlib.metadata, 'entry_points', lambda **kwargs: [
        SimpleNamespace(name='aml'), SimpleNamespace(name='AML')])
    with pytest.raises(RuntimeError, match='Ambiguous'):
        runner.validator('utility', 'AML')


def test_missing_reports_and_outside_assets_fail(tmp_path):
    with pytest.raises(RuntimeError, match='HTML report'):
        runner.standalone_report(tmp_path)
    (tmp_path / 'report.html').write_text('<img src="../private.png">')
    with pytest.raises(RuntimeError, match='unavailable'):
        runner.standalone_report(tmp_path)


def test_plugin_failure_propagates_before_completion(tmp_path, monkeypatch):
    (tmp_path / 'configuration.json').write_text(json.dumps({'metadata': {'variables': [{'name': 'age'}]},
        'evaluation': {}, 'use_case': 'SCD'}))
    for name in ('real', 'holdout', 'synthetic'):
        pd.DataFrame({'age': [1, 2]}).to_pickle(tmp_path / (name + '.pkl'))
    monkeypatch.setattr(runner, 'validator', lambda *args: Mock(side_effect=RuntimeError('evaluation failed')))
    with pytest.raises(RuntimeError, match='evaluation failed'):
        runner.evaluate(tmp_path)
    assert not (tmp_path / 'results.json').exists()
