from unittest.mock import MagicMock, patch

from flwr.server import ServerApp

from fl_server.utils import app_utils


def test_run_server_app():
    server_app = MagicMock(ServerApp)
    superlink_url = "url"
    root_certificates = b"certs"

    with (
        patch.object(app_utils, "update_console_handler") as mock_update_handler,
        patch.object(app_utils, "GrpcDriver") as mock_driver,
    ):
        app_utils.run_server_app(server_app, superlink_url, root_certificates)

    mock_update_handler.assert_called_once_with(
        level=app_utils.INFO, timestamps=True, colored=True
    )
    mock_driver.assert_called_once_with(
        driver_service_address=superlink_url, root_certificates=root_certificates
    )
    server_app.assert_called_once()
    mock_driver.return_value.close.assert_called_once()
