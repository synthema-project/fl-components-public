"""Exercise both identical lifecycle implementations without external services."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

ROOT = Path(__file__).resolve().parents[3]

@pytest.fixture(params=[
    ROOT / "common/interfaces/mlflow_client.py",
    ROOT / "apps/fl_server/src/fl_server/interfaces/mlflow_client.py",
])
def lifecycle(request):
    tree = ast.parse(request.param.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "clean_current_config")
    state = dict(
        _configured=SimpleNamespace(value=True),
        _current_config=SimpleNamespace(child_run_id="child", parent_run_id="parent", is_central_node=True),
        mlflow=Mock(), _mlflow_client=Mock(),
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(request.param), "exec"), state)
    return state

def test_failed_then_successful_run(lifecycle):
    s = lifecycle
    s["clean_current_config"]("FAILED")
    assert not s["_configured"].value
    assert s["_mlflow_client"].set_terminated.call_args_list[-1].kwargs["status"] == "FAILED"
    s["_configured"].value = True
    s["clean_current_config"]()
    assert not s["_configured"].value
    assert s["_mlflow_client"].set_terminated.call_args_list[-1].kwargs["status"] == "FINISHED"

def test_cleanup_failure_still_releases_state(lifecycle):
    s = lifecycle
    s["mlflow"].end_run.side_effect = RuntimeError("tracking unavailable")
    with pytest.raises(RuntimeError, match="tracking unavailable"):
        s["clean_current_config"]("FAILED")
    assert not s["_configured"].value
    assert s["_mlflow_client"].set_terminated.call_count == 2

def test_repeat_cleanup_does_not_rewrite_old_run(lifecycle):
    s = lifecycle
    s["clean_current_config"]("FAILED")
    s["_mlflow_client"].reset_mock()
    s["clean_current_config"]()
    s["_mlflow_client"].set_terminated.assert_not_called()

def test_partial_configuration_ends_active_run(lifecycle):
    s = lifecycle
    s["_configured"].value = False
    s["clean_current_config"]("FAILED")
    s["mlflow"].end_run.assert_called_once_with(status="FAILED")

def test_server_preserves_training_error_and_allows_next_task():
    import asyncio
    from unittest.mock import patch, AsyncMock
    from fl_server import app
    task = SimpleNamespace(id=1, use_case="AML", selected_nodes=["ICH"],
                           SMPC_enabled=False, dp_mode=None, num_global_iterations=1,
                           num_synthetic_samples=0)
    driver = Mock()
    with patch.object(app.requires, "filter_clients", return_value=[1]), \
         patch.object(app, "configure_mlflow", new_callable=AsyncMock), \
         patch.object(app, "load_model_and_data", new_callable=AsyncMock), \
         patch.object(app, "execute_training_loop", new_callable=AsyncMock) as train, \
         patch.object(app.requires, "get_parameters_from_one_node"), \
         patch.object(app.requires, "set_parameters"), \
         patch.object(app.requires, "upload_model") as upload, \
         patch.object(app.requires, "clean_config") as clean_clients, \
         patch.object(app.mlflow_client, "clean_current_config") as clean_server:
        train.side_effect = ValueError("training failed")
        clean_clients.side_effect = TimeoutError("client disconnected")
        with pytest.raises(ValueError, match="training failed"):
            asyncio.run(app.server_main(task, driver, Mock(), Mock()))
        clean_server.assert_called_once_with("FAILED")
        upload.assert_not_called()
        clean_clients.side_effect = None
        train.side_effect = None
        asyncio.run(app.server_main(task, driver, Mock(), Mock()))
        upload.assert_called_once()
        assert clean_server.call_count == 2
        clean_server.assert_called_with()

def test_cleanup_reply_wait_is_bounded():
    from unittest.mock import patch
    from fl_server.utils import driver_utils
    driver = Mock()
    driver.pull_messages.return_value = []
    with patch.object(driver_utils.time, "monotonic", side_effect=[0, 16]):
        with pytest.raises(TimeoutError, match="cleanup"):
            driver_utils.wait_messages(driver, ["missing"], timeout=15)
