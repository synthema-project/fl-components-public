import json
import pytest
from fl_server.schemas.task import TaskRead

BASE = dict(id=1, user_id="test", use_case="AML", model_name="gmm",
            model_version=1, num_global_iterations=1, run_name="test", experiment_name="test")

@pytest.mark.parametrize("fields, expected", [
    ({"n_samples_synthetic": 25}, 25),
    ({"num_synthetic_samples": 12}, 12),
    ({"n_samples_synthetic": 0}, 0),
    ({}, 10000),
])
@pytest.mark.parametrize("parse", [
    TaskRead.model_validate,
    lambda payload: TaskRead.model_validate_json(json.dumps(payload)),
])
def test_dispatch_sample_count(fields, expected, parse):
    assert parse({**BASE, **fields}).num_synthetic_samples == expected

@pytest.mark.parametrize("fields", [
    {"n_samples_synthetic": -1},
    {"n_samples_synthetic": True},
    {"n_samples_synthetic": 1.5},
    {"n_samples_synthetic": 5, "num_synthetic_samples": 6},
])
@pytest.mark.parametrize("parse", [
    TaskRead.model_validate,
    lambda payload: TaskRead.model_validate_json(json.dumps(payload)),
])
def test_invalid_count_rejected(fields, parse):
    with pytest.raises(ValueError):
        parse({**BASE, **fields})

@pytest.mark.asyncio
async def test_rabbitmq_parser_preserves_requested_generation():
    from fl_server.main import _parse_message
    task = await _parse_message(json.dumps({**BASE, "n_samples_synthetic": 10000}))
    assert task is not None
    assert task.num_synthetic_samples == 10000

