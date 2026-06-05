"""Validation test for Phase 3: DCAT-AP dataset_meta_json field in TaskBase."""
from fl_server.schemas.task import TaskBase


def test_task_base_has_dataset_meta_json():
    """Verify the dataset_meta_json field exists and is optional in TaskBase."""
    task = TaskBase(
        user_id="user_001",
        use_case="SCD",
        model_name="scd_model",
        model_version=1,
        num_global_iterations=10,
        run_name="run_001",
        experiment_name="SCD_experiment",
    )
    assert task.dataset_meta_json is None


def test_task_base_accepts_dataset_meta_json():
    """Verify the dataset_meta_json field accepts a serialised DCAT-AP payload."""
    dcat_payload = '{"dct:title": "AML Dataset", "dct:format": "text/csv"}'
    task = TaskBase(
        user_id="user_001",
        use_case="AML",
        model_name="aml_model",
        model_version=2,
        num_global_iterations=5,
        run_name="run_aml_001",
        experiment_name="AML_experiment",
        dataset_meta_json=dcat_payload,
    )
    assert task.dataset_meta_json == dcat_payload


def test_task_base_fields():
    """Verify all expected fields are present in TaskBase."""
    fields = set(TaskBase.model_fields.keys())
    assert "dataset_meta_json" in fields
    assert "user_id" in fields
    assert "use_case" in fields
    assert "model_name" in fields
    assert "model_version" in fields
