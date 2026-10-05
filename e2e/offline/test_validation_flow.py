"""Exercise real Flower dispatch through training, pool storage and report registration.

Only network services and evaluators are replaced; all records, stage transitions,
partition preparation and pool object serialization execute normally.
"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np
from flwr.common import Context, Error, Message, Metadata, RecordSet
from testing import configuration, create_partitions, model
from interfaces.dataset_snapshot import select_snapshot
from fl_server import app as server, requires
from fl_server.schemas.task import TaskRead
from fl_client import main as client, stages as client_stages
from interfaces import mlflow_client as client_mlflow
from fl_client import validation_runner

class LocalDriver:
    def __init__(self):
        self.replies = {}
        self.modes = []
    def get_node_ids(self):
        return [9876]
    def create_message(self, content, message_type, dst_node_id, group_id, ttl):
        return Message(content=content, metadata=Metadata(1,str(len(self.modes)),0,dst_node_id,'',group_id,ttl,message_type))
    def push_messages(self, messages):
        for msg in messages:
            mode = msg.content.configs_records['config'].get('mode', msg.metadata.message_type)
            self.modes.append(mode)
            try:
                handler = client.train if msg.metadata.message_type == 'train' else client.query
                response = handler(msg, Context(state=RecordSet()))
            except Exception as error:
                response = msg.create_error_reply(Error(2, str(error)))
            self.replies[msg.metadata.message_id] = response
            yield msg.metadata.message_id
    def pull_messages(self, message_ids):
        return [self.replies.pop(i) for i in message_ids]



def test_frozen_full_validation_flow(tmp_path, monkeypatch):
    monkeypatch.setenv('VALIDATION_PROFILE', 'aml-full')
    metadata, evaluation = configuration()
    paths = create_partitions(tmp_path)
    import hashlib
    version = hashlib.sha256(json.dumps({'metadata': metadata, 'evaluation': evaluation},
                            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    records = [{'node': 'ICH', 'collection_id': 'one', 'configuration_version': version,
                'use_case': 'AML', 'path': path, 'dataset_role': role,
                'dataset_meta': metadata, 'evaluation_configuration': evaluation}
                for role, path in paths.items()]
    snapshot = select_snapshot(records, 'AML', ['ICH'])
    task = TaskRead(user_id='test', id=1, use_case='AML', selected_nodes=['ICH'],
        model_name='test', model_version=1, num_global_iterations=2, run_name='offline',
        experiment_name='offline', num_synthetic_samples=25, validation_snapshot=snapshot)
    driver = LocalDriver()
    client.global_vars.update(node_name='ICH', use_case='AML', model_meta=None, SMPC_enabled=False, dp_config=None)
    m = model()
    def evaluator(**kwargs):
        assert len(kwargs['real_data']) == 72
        assert len(kwargs['holdout_data']) == 24
        assert len(kwargs['synth_data']) == 25
        (Path(kwargs['output_folder']) / 'report.html').write_text('<html>Artificial evaluator result</html>')
    def subprocess_evaluation(args, **kwargs):
        validation_runner.evaluate(args[-1])
    objects = {}
    def put(bucket, key, stream, **kwargs):
        objects[(bucket, key)] = stream.getvalue()
    def get(bucket, key, target):
        Path(target).write_bytes(objects[(bucket, key)])
    with patch.object(server.mlflow_utils, 'create_mlflow_runs', return_value=('1', 'parent', 'child')), \
         patch.object(server.mlflow_client, 'set_current_config'), \
         patch.object(server.mlflow_client, 'load_model', return_value=m), \
         patch.object(server.mlflow_client, 'log_metrics'), \
         patch.object(server.mlflow_client, 'clean_current_config'), \
         patch.object(client_stages, 'set_run_config'), \
         patch.object(client_mlflow, 'load_model', return_value=m), \
         patch.object(client_mlflow, 'load_dataset_meta', return_value=None), \
         patch.object(client_mlflow, 'log_metrics'), \
         patch.object(client_mlflow, 'clean_current_config'), \
         patch.object(client_mlflow, 'upload_final_state', return_value={'model_id': 'trained', 'run_id': 'child'}) as upload, \
         patch('fl_client.validation_runner.validator', return_value=evaluator), \
         patch('fl_client.validation_runner.subprocess.run', side_effect=subprocess_evaluation), \
         patch.object(requires, 'DataCatalogueClient') as catalogue, \
         patch('fl_client.utils.minio_loader._get_minio_client') as minio:
        catalogue.return_value.register_synthetic_pool.side_effect = lambda **kwargs: {'pool_id': kwargs['pool_id']}
        minio.return_value.bucket_exists.return_value = True
        minio.return_value.get_bucket_policy.return_value = '{"Statement": []}'
        minio.return_value.put_object.side_effect = put
        minio.return_value.fget_object.side_effect = get
        asyncio.run(server.server_main(task, driver, Context(state=RecordSet()), MagicMock()))
        catalogue.return_value.get_use_case_datasets.assert_not_called()
        upload.assert_called_once()
        assert np.any(upload.call_args.args[0].mean != 0)
        registration = catalogue.return_value.register_synthetic_pool.call_args.kwargs
        assert registration['dataset_snapshot'] == snapshot
        assert registration['validation_nodes'] == ['ICH']
        assert registration['n_samples'] == 25
        assert catalogue.return_value.register_pool_validation.call_count == 3
        assert {call.kwargs['metrics']['scope'] for call in catalogue.return_value.register_pool_validation.call_args_list} == {'utility', 'privacy', 'fidelity'}
        csv = minio.return_value.put_object.call_args_list[2]
        assert csv.kwargs['tags']['validation_status'] == 'pending'
        assert minio.return_value.put_object.call_count == 6
        assert all(call.kwargs['report_uri'].startswith('s3://synthetic-validation-reports/') for call in catalogue.return_value.register_pool_validation.call_args_list)
        assert 'validate_synthetic_pool' in driver.modes
