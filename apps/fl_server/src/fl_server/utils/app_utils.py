from typing import Optional
from logging import INFO

from flwr.common import event, EventType, Context, RecordSet
from flwr.common.typing import UserConfig
from flwr.server.driver import GrpcDriver
from flwr.server import ServerApp
from flwr.common.logger import update_console_handler


def run_server_app(
    server_app: ServerApp,
    superlink_url: str,
    root_certificates: Optional[bytes] = None,
    node_config: UserConfig = dict(),
    run_config: UserConfig = dict(),
) -> None:
    update_console_handler(
        level=INFO,
        timestamps=True,
        colored=True,
    )

    driver = GrpcDriver(
        run_id=0,
        driver_service_address=superlink_url,
        root_certificates=root_certificates,
    )

    context = Context(
        node_id=0, node_config=node_config, state=RecordSet(), run_config=run_config
    )
    server_app(driver, context)

    driver.close()

    event(EventType.RUN_SERVER_APP_LEAVE)
