from unittest.mock import Mock
from flwr.common import Message, Metadata, RecordSet
from fl_client.utils.responses_utils import create_participate_response


def test_selection_response_identifies_client_even_when_ineligible():
    for participate in (True, False):
        request = Message(metadata=Mock(spec=Metadata), content=RecordSet())
        reply = create_participate_response(request, participate, 'ICH')
        config = reply.content.configs_records['config']
        assert config['node_name'] == 'ICH'
        assert config['participate'] is participate
        assert config['success'] is True
