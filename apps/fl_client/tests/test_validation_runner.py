"""FL reporting delegates to MSI's Flower simulation helpers and input roles."""
import json
from pathlib import Path
import pandas as pd
import pytest
from fl_client import validation_runner as runner
from model_store_interface import validation
from model_store_interface.metadata.models.model import SynthemaDatasetModel


def setup_work(work):
    model = SynthemaDatasetModel.from_json(
        {'title': 'Test AML', 'use_case': 'AML', 'variables': [
            {'name': 'age', 'datatype': 'Text', 'description': 'Test feature'}]},
        {'utility_evaluation': {'clinical_variables': ['age'], 'genomic_variables': []},
         'privacy_evaluation': {'known_attributes': ['age'], 'sensitive_attributes': [],
                                'differential_privacy': False}})
    (work / 'configuration.json').write_text(json.dumps({
        'dataset_meta': model.model_dump(mode='json'), 'node_id': 'ICH'}))
    # Validators receive original feature names matching the MSI metadata.
    for role, values in [('train', [1, 2]), ('test', [3, 4]), ('synthetic', [5, 6])]:
        pd.DataFrame({'age': values}).to_pickle(work / f'{role}.pkl')


def test_progress_monitor_logs_stages_and_elapsed_time_without_raw_output(tmp_path, caplog):
    import threading
    runner.progress(tmp_path, 'utility', 'started')
    runner.progress(tmp_path, 'utility', 'completed')
    runner.progress(tmp_path, 'privacy', 'started')
    # A stopped monitor still drains final events when the process exits quickly.
    stop = threading.Event()
    stop.set()
    runner.monitor_progress(tmp_path, stop, heartbeat=0)
    assert 'utility started' in caplog.text
    assert 'utility completed' in caplog.text
    assert 'privacy started' in caplog.text
    assert 'last stage=privacy' in caplog.text
    assert 'elapsed' in caplog.text


def test_progress_monitor_retries_incomplete_event(tmp_path, caplog):
    import threading
    runner.progress(tmp_path, 'utility', 'started')
    with (tmp_path / 'progress.jsonl').open('a') as stream:
        stream.write('{"stage":')
    stop = threading.Event()
    stop.set()
    runner.monitor_progress(tmp_path, stop)
    assert 'utility started' in caplog.text


def test_stack_diagnostics_include_locations_without_local_values(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    frame = SimpleNamespace(f_code=SimpleNamespace(co_filename='/private/fitting.py', co_name='fit'),
                            f_lineno=42, f_back=None, f_locals={'patient': 'patient-secret'})
    monkeypatch.setattr(runner.sys, '_current_frames', lambda: {-1: frame})
    stop = Mock()
    stop.wait.side_effect = [False, True]
    runner.monitor_evaluator_stacks(tmp_path, stop)
    text = (tmp_path / 'progress.jsonl').read_text()
    assert 'fitting.py:42 (fit)' in text
    assert 'patient-secret' not in text
    assert '/private/' not in text


def test_process_snapshot_finds_workers_created_by_threads(tmp_path):
    def process(pid, state, cpu, child=''):
        directory = tmp_path / str(pid)
        (directory / 'task' / str(pid + 1)).mkdir(parents=True)
        fields = ['0'] * 22
        fields[0], fields[11], fields[12] = state, str(cpu), '5'
        fields[17], fields[19], fields[21] = '4', str(pid), '256'
        (directory / 'stat').write_text(f'{pid} (name with ) spaces) ' + ' '.join(fields))
        (directory / 'task' / str(pid + 1) / 'children').write_text(child)
    process(100, 'S', 20, '200 999')  # 999 exited before it could be sampled.
    process(200, 'R', 40)
    snapshot = runner.process_snapshot(100, proc=tmp_path)
    assert set(snapshot) == {100, 200}
    assert snapshot[100]['state'] == 'S'
    assert snapshot[200]['state'] == 'R'
    assert snapshot[100]['threads'] == 4
    assert snapshot[100]['cpu'] == 25 / runner.os.sysconf('SC_CLK_TCK')
    assert snapshot[100]['rss_mib'] > 0


def test_parent_diagnostics_report_cpu_delta_even_without_child_progress(tmp_path, monkeypatch, caplog):
    (tmp_path / 'evaluator.pid').write_text('100')
    old = {100: {'state': 'S', 'cpu': 5, 'threads': 4, 'rss_mib': 20, 'start': '1'}}
    current = {100: dict(old[100], cpu=5),
               200: {'state': 'R', 'cpu': 10, 'threads': 2, 'rss_mib': 30, 'start': '2'}}
    monkeypatch.setattr(runner, 'process_snapshot', lambda pid: current)
    assert runner.log_process_activity(tmp_path, old, 30) == current
    assert 'role=evaluator state=S' in caplog.text
    assert 'cpu_delta_s=0.00' in caplog.text
    assert 'role=worker state=R' in caplog.text
    assert 'first sample' in caplog.text


def test_stack_monitor_failure_is_visible_without_exception_values(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError('patient-secret')
    monkeypatch.setattr(runner, 'sample_evaluator_stacks', fail)
    runner.monitor_evaluator_stacks(tmp_path, None)
    text = (tmp_path / 'progress.jsonl').read_text()
    assert 'failed (RuntimeError)' in text
    assert 'patient-secret' not in text


def test_msi_helpers_receive_exact_flower_partition_roles_and_unmodified_html(tmp_path, monkeypatch):
    setup_work(tmp_path)
    monkeypatch.delenv('FIDELITY_ENABLED', raising=False)
    calls = {}
    html = '<html><img src="missing.png"><a href="../details.html">Details</a></html>'
    def helper(kind):
        def run(**kwargs):
            calls[kind] = kwargs
            path = Path(kwargs['output_folder']) / 'report.html'
            path.write_text(html)
            return kwargs['output_folder'] if kind == 'utility' else str(path)
        return run
    monkeypatch.setattr(validation, 'validate_utility_report', helper('utility'))
    monkeypatch.setattr(validation, 'validate_privacy_report', helper('privacy'))
    runner.evaluate(tmp_path)
    assert calls['utility']['real_data'].age.tolist() == [3, 4]
    assert calls['utility']['holdout_data'].empty
    assert list(calls['utility']['holdout_data']) == ['age']
    assert calls['privacy']['real_data'].age.tolist() == [1, 2]
    assert calls['privacy']['holdout_data'].age.tolist() == [3, 4]
    assert calls['utility']['synthetic_data'].age.tolist() == [5, 6]
    assert calls['utility']['node_id'] == 'ICH'
    assert isinstance(calls['utility']['dataset_meta'], SynthemaDatasetModel)
    assert (tmp_path / 'utility.html').read_text() == html
    assert (tmp_path / 'privacy.html').read_text() == html


def test_optional_val_partition_matches_simulation(tmp_path, monkeypatch):
    setup_work(tmp_path)
    pd.DataFrame({'age': [7, 8]}).to_pickle(tmp_path / 'val.pkl')
    received = []
    def helper(**kwargs):
        received.append(kwargs['holdout_data'].age.tolist())
        path = Path(kwargs['output_folder']) / 'report.html'
        path.write_text('<html>report</html>')
        return str(path)
    monkeypatch.setattr(validation, 'validate_utility_report', helper)
    monkeypatch.setattr(validation, 'validate_privacy_report', helper)
    runner.evaluate(tmp_path)
    assert received == [[7, 8], [3, 4]]


def test_all_evaluator_inputs_follow_metadata_order_without_changing_values(tmp_path, monkeypatch):
    setup_work(tmp_path)
    config = json.loads((tmp_path / 'configuration.json').read_text())
    config['dataset_meta']['metadata']['variables'].append(
        {'name': 'sex', 'datatype': 'Binary', 'description': 'Sex',
         'categories': ['0', '1'], 'mapping': {'0': 0, '1': 1}})
    (tmp_path / 'configuration.json').write_text(json.dumps(config))
    originals = {}
    for role in ('train', 'test', 'synthetic', 'val'):
        originals[role] = pd.DataFrame({'sex': ['0', '1'], role + '_extra': [9, 9],
                                        'age': [21, 42]})
        originals[role].to_pickle(tmp_path / f'{role}.pkl')
    calls = []
    def helper(**kwargs):
        calls.append(kwargs)
        for key in ('real_data', 'holdout_data', 'synthetic_data'):
            pd.testing.assert_frame_equal(
                kwargs[key], pd.DataFrame({'age': [21, 42], 'sex': ['0', '1']}))
        path = Path(kwargs['output_folder']) / 'report.html'
        path.write_text('<html>report</html>')
        return str(path)
    monkeypatch.setattr(validation, 'validate_utility_report', helper)
    monkeypatch.setattr(validation, 'validate_privacy_report', helper)
    runner.evaluate(tmp_path)
    assert len(calls) == 2
    for role, original in originals.items():
        pd.testing.assert_frame_equal(pd.read_pickle(tmp_path / f'{role}.pkl'), original)


@pytest.mark.parametrize('role', ['train', 'test', 'synthetic', 'val'])
def test_missing_metadata_columns_fail_before_evaluation(tmp_path, monkeypatch, role):
    setup_work(tmp_path)
    pd.DataFrame({'extra': [1, 2]}).to_pickle(tmp_path / f'{role}.pkl')
    def unexpected(**kwargs):
        pytest.fail('Evaluator must not run with missing metadata columns')
    monkeypatch.setattr(validation, 'validate_utility_report', unexpected)
    monkeypatch.setattr(validation, 'validate_privacy_report', unexpected)
    with pytest.raises(ValueError, match=f'{role} partition.*age'):
        runner.evaluate(tmp_path)


def test_original_msi_fallback_is_preserved_and_identified(tmp_path, monkeypatch):
    setup_work(tmp_path)
    monkeypatch.setattr(validation, '_discover_plugin', lambda *args: None)
    runner.evaluate(tmp_path)
    results = json.loads((tmp_path / 'results.json').read_text())
    assert results == [{'kind': 'utility', 'fallback': True}, {'kind': 'privacy', 'fallback': True}]
    assert 'preliminary validation report' in (tmp_path / 'utility.html').read_text()


def test_subprocess_diagnostics_do_not_include_patient_values(tmp_path, monkeypatch):
    def fail(work):
        (work / 'stage.json').write_text(json.dumps({'kind': 'privacy'}))
        raise ValueError('patient-secret-value')
    monkeypatch.setattr(runner, 'evaluate', fail)
    with pytest.raises(ValueError):
        runner.run_evaluation(tmp_path)
    text = (tmp_path / 'failure.json').read_text()
    assert json.loads(text)['kind'] == 'privacy'
    assert 'patient-secret-value' not in text


def test_publication_uses_original_partitions_and_original_report_content(tmp_path, monkeypatch):
    from fl_client.utils import minio_loader
    synthetic = tmp_path / 'synthetic.csv'
    pd.DataFrame({'age': [5, 6]}).to_csv(synthetic, index=False)
    setup_work(tmp_path)
    metadata = SynthemaDatasetModel.model_validate(
        json.loads((tmp_path / 'configuration.json').read_text())['dataset_meta'])
    monkeypatch.setenv('VALIDATION_PROFILE', 'aml-full')
    monkeypatch.setattr(minio_loader, 'download_object', lambda _: str(synthetic))
    uploads = []
    def upload(content, **kwargs):
        uploads.append((content, kwargs))
        return 's3://private/report.html'
    monkeypatch.setattr(minio_loader, 'upload_pool_report', upload)
    def subprocess_run(command, **kwargs):
        work = Path(command[-1])
        assert pd.read_pickle(work / 'train.pkl').age.tolist() == [1, 2]
        assert pd.read_pickle(work / 'test.pkl').age.tolist() == [3, 4]
        (work / 'utility.html').write_text('<html>Original MSI report</html>')
        (work / 'results.json').write_text(json.dumps([{'kind': 'utility', 'fallback': True}]))
    monkeypatch.setattr(runner.subprocess, 'run', subprocess_run)
    state = {'model_meta': metadata, 'node_name': 'ICH',
             'prepared_partitions': {role: pd.DataFrame({'encoded_feature': [100, 200]})
                                     for role in ('train', 'test')},
             'data_partitions': {role: pd.DataFrame({'age': values})
                                 for role, values in [('train', [1, 2]), ('test', [3, 4])]}}
    result = runner.generate_reports('pool-id', 's3://pool/data.csv', 'AML', state)
    assert uploads[0][0] == '<html>Original MSI report</html>'
    assert uploads[0][1]['node_name'] == 'ICH'
    assert result['reports'][0]['metrics']['fallback'] is True
    assert result['reports'][0]['metrics']['privacy_evaluated'] is False


def test_msi_caught_plugin_failure_is_reported_without_exception_values(tmp_path, monkeypatch):
    setup_work(tmp_path)
    def fail(**kwargs):
        raise ValueError('private-patient-value')
    monkeypatch.setattr(validation, '_discover_plugin', lambda *args: fail)
    runner.evaluate(tmp_path)
    text = (tmp_path / 'results.json').read_text()
    results = json.loads(text)
    assert all(result['fallback'] for result in results)
    assert results[0]['evaluator_failures'][0]['type'] == 'ValueError'
    assert 'fail' in results[0]['evaluator_failures'][0]['location']
    assert 'private-patient-value' not in text
