import pytest
from unittest.mock import MagicMock, patch
from flwr.server import ServerApp
from fl_server.utils import app_utils


@pytest.mark.asyncio
async def test_run_server_app():
    """Tests that `run_server_app` initializes and runs the FL server correctly."""
    mock_server_app = MagicMock(spec=ServerApp)  # ✅ Mock ServerApp
    superlink_url = "url"
    root_certificates = b"certs"

    with (
        patch.object(app_utils, "update_console_handler") as mock_update_handler,
        patch.object(app_utils, "GrpcDriver") as mock_driver,
    ):
        mock_driver_instance = mock_driver.return_value  # ✅ Ensure instance mock
        await app_utils.run_server_app(
            mock_server_app, superlink_url, root_certificates
        )

    # ✅ Verify that logging setup was correctly initialized
    mock_update_handler.assert_called_once_with(
        level=app_utils.INFO, timestamps=True, colored=True
    )

    # ✅ Ensure `GrpcDriver` is instantiated correctly
    mock_driver.assert_called_once_with(
        driver_service_address=superlink_url, root_certificates=root_certificates
    )

    # ✅ Ensure `ServerApp` was called instead of `.run()`
    mock_server_app.assert_called_once_with(
        mock_driver_instance, app_utils.Context(state=app_utils.RecordSet())
    )

    # ✅ Ensure the driver was closed properly
    mock_driver_instance.close.assert_called_once()
