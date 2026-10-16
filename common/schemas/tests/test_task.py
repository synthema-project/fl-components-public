import pytest

from schemas.task import Status, TaskRead


@pytest.fixture
def task():
    return TaskRead(
        user_id="user",
        model_name="model",
        model_version=1,
        num_global_iterations=10,
        use_case="use_case",
        id=1,
        created_at="2021-01-01T00:00:00",
        status="pending",
        run_name="run_name",
        experiment_name="experiment_name",
    )


def test_task_serde(task):
    js = task.model_dump_json()

    new_task = TaskRead.model_validate_json(js)

    new_task_dict = new_task.model_dump()
    task_dict = new_task.model_dump()
    assert new_task_dict == task_dict


def test_task_change_status(task):
    task_db = TaskRead.model_validate(task)

    task_db.change_status("RUNNING")
    assert task_db.status == Status.RUNNING
    task_db.change_status("cancelled")
    assert task_db.status == Status.CANCELLED


def test_smpc_enabled_without_platform_dp_by_default(task):
    from schemas.task import TaskCreate
    request = TaskCreate.model_validate(task.model_dump())
    assert request.dp_mode is None
    assert request.SMPC_enabled is True
    assert request.noise_multiplier is None
    assert request.clipping_norm is None


@pytest.mark.parametrize('mode', [None, 'central_dp', 'local_dp', 'None'])
def test_original_task_dp_modes(task, mode):
    request = TaskRead.model_validate({**task.model_dump(), 'dp_mode': mode, 'SMPC_enabled': False})
    assert request.dp_mode == (None if mode == 'None' else mode)


def test_central_dp_smpc_combination_rejected(task):
    mode = 'central_dp'
    with pytest.raises(ValueError, match='SMPC cannot be enabled'):
        TaskRead.model_validate({**task.model_dump(), 'dp_mode': mode, 'SMPC_enabled': True})


def test_model_local_dp_allowed_with_smpc(task):
    request = TaskRead.model_validate({**task.model_dump(), 'dp_mode': 'local_dp', 'SMPC_enabled': True})
    assert request.SMPC_enabled is True
    assert request.dp_mode == 'local_dp'


def test_default_created_at_matches_existing_postgres_timestamp(task):
    from datetime import datetime, timezone
    from unittest.mock import patch

    instant = datetime(2026, 10, 16, 13, 22, 43, 516492, tzinfo=timezone.utc)
    # Exercise the same model validation path as POST /tasks/ with no timestamp.
    with patch('schemas.task.datetime') as clock:
        clock.now.return_value = instant
        created = TaskRead.model_validate(task.model_dump(exclude={'created_at'}))
    clock.now.assert_called_once_with(timezone.utc)
    assert created.created_at == instant.replace(tzinfo=None)
    assert created.created_at.tzinfo is None
    assert TaskRead.__table__.c.created_at.type.timezone is False
    import json
    assert json.loads(created.model_dump_json())['created_at'] == '2026-10-16T13:22:43.516492'
