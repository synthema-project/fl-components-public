"""Queued tasks retain immutable configuration and complete dataset collections."""
import hashlib
import json
import pytest
from interfaces.dataset_snapshot import select_snapshot
from schemas.task import TaskCreate, TaskRead
from sqlalchemy import create_engine
from sqlmodel import SQLModel, Session


def records(node='ICH', collection='one', stamp='2026-01-01'):
    docs = {'metadata': {'variables': [{'name': 'age'}]}, 'evaluation': {'clinical': ['age']}}
    version = hashlib.sha256(json.dumps(docs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return [{'node': node, 'collection_id': collection, 'timestamp': stamp, 'use_case': 'AML',
             'dataset_role': role, 'path': f'{collection}/{role}', 'configuration_version': version,
             'dataset_meta': docs['metadata'], 'evaluation_configuration': docs['evaluation']}
            for role in ('train', 'test', 'schema')]


def test_new_incomplete_upload_cannot_mix_with_old_partitions():
    rows = records() + records(collection='two', stamp='2026-02-01')[:2]
    snap = select_snapshot(rows, 'AML', ['ICH'])
    assert snap['nodes']['ICH']['collection_id'] == 'one'
    rows += records(collection='two', stamp='2026-02-01')[2:]
    assert select_snapshot(rows, 'AML', ['ICH'])['nodes']['ICH']['collection_id'] == 'two'
    assert snap['nodes']['ICH']['paths']['test'] == 'one/test'


def test_configuration_tampering_and_version_mismatch_fail():
    rows = records(); rows[0]['configuration_version'] = 'changed'
    with pytest.raises(ValueError, match='inconsistent'):
        select_snapshot(rows, 'AML', ['ICH'])
    rows = records()
    for row in rows:
        row['dataset_meta'] = {'changed': True}
    with pytest.raises(ValueError, match='content'):
        select_snapshot(rows, 'AML', ['ICH'])
    with pytest.raises(ValueError, match='complete'):
        select_snapshot(records(), 'AML', ['ICH', 'UMCU'])


def test_task_snapshot_survives_database_and_queue_roundtrip():
    snapshot = select_snapshot(records(), 'AML', ['ICH'])
    engine = create_engine('sqlite://'); SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        task = TaskRead.model_validate(TaskCreate(user_id='test', use_case='AML', model_name='model',
            model_version=1, num_global_iterations=1, run_name='test', experiment_name='test',
            selected_nodes=['ICH'], validation_snapshot=snapshot))
        session.add(task); session.commit(); session.refresh(task)
        restored = TaskRead.model_validate_json(task.model_dump_json())
        assert restored.validation_snapshot == snapshot


def test_two_part_collection_ignores_legacy_val_and_requires_test():
    rows = records()
    rows.append({**rows[0], 'dataset_role': 'val', 'configuration_version': 'obsolete'})
    assert set(select_snapshot(rows, 'AML', ['ICH'])['nodes']['ICH']['paths']) == {'train', 'test', 'schema'}
    with pytest.raises(ValueError, match='complete'):
        select_snapshot([row for row in rows if row['dataset_role'] != 'test'], 'AML', ['ICH'])
