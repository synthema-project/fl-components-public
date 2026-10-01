"""Flower reporting contract: all report types are JSON encoded with success status."""
import json
from unittest.mock import Mock, patch
from flwr.common import Message, Metadata, RecordSet
from interfaces.recordset import create_validate_synthetic_pool_recordset
from fl_client import interface


def test_full_report_reply_contains_all_reports_and_success():
    msg = Message(metadata=Mock(spec=Metadata), content=create_validate_synthetic_pool_recordset(
        'pool', 's3://pools/AML/pool/data.csv', 'AML'))
    reports = [{'report_uri': f's3://pools/AML/pool/evaluation_reports/ICH_{kind}.html',
                'metrics': {'scope': kind, 'privacy_evaluated': kind == 'privacy', 'fallback': False}}
               for kind in ('utility', 'privacy', 'fidelity')]
    with patch('fl_client.interface.stages.validate_synthetic_pool', return_value={'node_name': 'ICH', 'reports': reports}):
        response = interface.validate_synthetic_pool_if(msg, {'node_name': 'ICH'})
    assert response.content.configs_records['config']['success']
    assert json.loads(response.content.configs_records['validation']['reports_json']) == reports
