"""Kubernetes SMPC discovery without contacting a cluster."""
from unittest.mock import Mock, mock_open, patch

import pytest
from fl_client.utils import peer_discovery as discovery


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(discovery.PeerDiscovery, '_get_local_ip', lambda self: '10.0.0.1')
    monkeypatch.setattr(discovery, 'IS_KUBERNETES', True)
    monkeypatch.setattr(discovery, 'K8S_NAMESPACE', 'synthema-personal')
    monkeypatch.setattr(discovery, 'K8S_API_URL', 'https://kubernetes:443')
    monkeypatch.setattr(discovery, 'K8S_LABEL_SELECTOR', 'app=fl-chart,component=fl-client')
    monkeypatch.setattr(discovery.socket, 'gethostname', lambda: 'ich-pod')
    monkeypatch.setattr(discovery.os.path, 'exists', lambda path: True)
    return discovery.PeerDiscovery('ICH', 50051, min_peers=1, max_discovery_time=2)


def pod(name, ip, phase='Running'):
    return {'metadata': {'name': name}, 'status': {'phase': phase, 'podIP': ip}}


def test_discovers_other_running_client_in_actual_namespace(service, monkeypatch):
    response = Mock(status_code=200)
    response.json.return_value = {'items': [pod('ich-pod', '10.0.0.1'),
        pod('umcu-pod', '10.0.0.2'), pod('pending-pod', '10.0.0.3', 'Pending')]}
    monkeypatch.setattr(discovery.time, 'sleep', lambda seconds: None)
    with patch('builtins.open', mock_open(read_data='test-token')), patch.object(
        discovery.requests, 'get', return_value=response,
    ) as get:
        service._k8s_discover_peers()
    assert service.peers == {'umcu-pod': '10.0.0.2:50051'}
    get.assert_called_once_with(
        'https://kubernetes:443/api/v1/namespaces/synthema-personal/pods',
        headers={'Authorization': 'Bearer test-token'},
        params={'labelSelector': 'app=fl-chart,component=fl-client'},
        verify='/var/run/secrets/kubernetes.io/serviceaccount/ca.crt', timeout=5)


@pytest.mark.parametrize('status', [401, 403])
def test_permissions_fail_immediately_with_actionable_error(service, status):
    with patch('builtins.open', mock_open(read_data='test-token')), patch.object(
        discovery.requests, 'get', return_value=Mock(status_code=status),
    ) as get:
        with pytest.raises(RuntimeError, match=f'synthema-personal.*HTTP {status}.*RoleBinding'):
            service.start_discovery()
    service.shutdown()
    get.assert_called_once()
    assert service.discovery_complete.is_set()


def test_namespace_uses_downward_api_or_service_account_file(monkeypatch):
    monkeypatch.setenv('POD_NAMESPACE', 'synthema-data')
    assert discovery.pod_namespace() == 'synthema-data'
    monkeypatch.delenv('POD_NAMESPACE')
    with patch.object(discovery.Path, 'exists', return_value=True), patch.object(
        discovery.Path, 'read_text', return_value='synthema-personal\n',
    ):
        assert discovery.pod_namespace() == 'synthema-personal'


def test_missing_service_account_fails_without_waiting_for_timeout(service, monkeypatch):
    monkeypatch.setattr(discovery.os.path, 'exists', lambda path: False)
    with pytest.raises(RuntimeError, match='mounted Kubernetes service account'):
        service.start_discovery()
    service.shutdown()
