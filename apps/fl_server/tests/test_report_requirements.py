from unittest.mock import Mock
import pytest
from fl_server.data_catalogue import DataCatalogueClient


@pytest.mark.parametrize('enabled, expected', [
    ('false', ['utility', 'privacy']),
    ('true', ['utility', 'privacy', 'fidelity']),
])
def test_pool_registration_freezes_enabled_reports(monkeypatch, enabled, expected):
    monkeypatch.setenv('FIDELITY_ENABLED', enabled)
    monkeypatch.setattr('fl_server.data_catalogue.catalogue_headers', lambda token: {})
    post = Mock()
    monkeypatch.setattr('fl_server.data_catalogue.requests.post', post)
    DataCatalogueClient('http://catalogue').register_synthetic_pool(
        pool_id='pool', use_case='AML', model_name='model', node_name='ICH',
        n_samples=10, object_uri='s3://pools/pool.csv', token='test')
    assert post.call_args.kwargs['json']['required_reports'] == expected
