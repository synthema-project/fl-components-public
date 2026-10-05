from types import SimpleNamespace
import pytest
from flwr.common import RecordSet, ConfigsRecord
from fl_server.utils.requires_utils import check_success_clients


def reply(error=None, config=None, content=True):
    return SimpleNamespace(metadata=SimpleNamespace(src_node_id=7),
        has_error=lambda: error is not None, error=error,
        has_content=lambda: content,
        content=RecordSet(configs_records={'config': ConfigsRecord(config or {})}))


def test_client_exception_preserves_original_error():
    with pytest.raises(RuntimeError, match='prepare_data failed'):
        check_success_clients([reply(error=SimpleNamespace(code=2,reason='prepare_data failed'))])


def test_explicit_failure_stops_workflow():
    with pytest.raises(RuntimeError, match='bad schema'):
        check_success_clients([reply(config={'success':False,'message':'bad schema'})])


def test_empty_reply_stops_workflow():
    with pytest.raises(RuntimeError, match='no content'):
        check_success_clients([reply(content=False)])


def test_success_accepted():
    check_success_clients([reply(config={'success':True})])
