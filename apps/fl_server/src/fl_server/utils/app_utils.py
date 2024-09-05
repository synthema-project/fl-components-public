from typing import Optional
from logging import INFO

from flwr.common import event, EventType, Context, RecordSet
from flwr.server.driver import GrpcDriver
from flwr.server import ServerApp
from flwr.common.logger import update_console_handler


def run_server_app(
    server_app: ServerApp,
    superlink_url: str,
    root_certificates: Optional[bytes] = None,
) -> None:
    update_console_handler(
        level=INFO,
        timestamps=True,
        colored=True,
    )

    driver = GrpcDriver(
        driver_service_address=superlink_url, root_certificates=root_certificates
    )

    context = Context(state=RecordSet())
    server_app(driver, context)

    driver.close()

    event(EventType.RUN_SERVER_APP_LEAVE)
