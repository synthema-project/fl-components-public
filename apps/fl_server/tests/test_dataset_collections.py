"""Train/test collections remain usable with or without historical val objects."""
from unittest.mock import Mock, patch
from fl_server.data_catalogue import DataCatalogueClient


def test_two_partition_collection_is_selected_without_legacy_val():
    rows = [{'node': 'UMCU', 'collection_id': 'one', 'dataset_role': role,
             'path': f'one/{role}', 'use_case': 'AML', 'configuration_version': 'v1',
             'dataset_meta': {'variables': []}, 'evaluation_configuration': {'utility': {}}}
            for role in ('train', 'test', 'schema')]
    rows.append({**rows[0], 'dataset_role': 'val', 'configuration_version': 'old'})
    response = Mock()
    response.json.return_value = {'datasets': rows}
    client = DataCatalogueClient('http://catalogue', token='test')
    with patch('fl_server.data_catalogue.catalogue_headers', return_value={}), patch(
        'fl_server.data_catalogue.requests.get', return_value=response,
    ):
        assert client.get_use_case_datasets('AML') == {
            'UMCU': {'train': 'one/train', 'test': 'one/test', 'schema': 'one/schema'}}
        response.json.return_value = {'datasets': [r for r in rows if r['dataset_role'] != 'test']}
        assert client.get_use_case_datasets('AML') == {}
