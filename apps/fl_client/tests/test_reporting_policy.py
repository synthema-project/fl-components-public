from types import SimpleNamespace
from fl_client.reporting_policy import utility_reporting_enabled


def test_pool_only_ignores_reporting_metadata(monkeypatch):
    monkeypatch.setenv('VALIDATION_PROFILE', 'none')
    monkeypatch.setenv('AML_UTILITY_PYTHON', '/irrelevant/python')
    assert not utility_reporting_enabled(SimpleNamespace(evaluation={'aml_utility': {'analyses': ['PCA']}}))


def test_reporting_requires_model_configuration(monkeypatch):
    monkeypatch.setenv('VALIDATION_PROFILE', 'aml')
    assert utility_reporting_enabled(SimpleNamespace(evaluation={'aml_utility': {'analyses': ['PCA']}}))
    assert not utility_reporting_enabled(SimpleNamespace(evaluation={}))
    assert not utility_reporting_enabled(None)
