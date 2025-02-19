import asyncio
import structlog
from status_listener.db import create_db_and_tables
from status_listener.status_consumer import consume_status_updates

logger = structlog.get_logger()


async def main() -> None:
    logger.info("Starting Status Listener Service...")
    await create_db_and_tables()

    while True:
        try:
            await consume_status_updates()
        except asyncio.CancelledError:
            logger.info("Status consumer cancelled, shutting down.")
            return
        except Exception as e:
            logger.error(f"Unexpected error in main consumer loop: {e}")
            await asyncio.sleep(5)  # Wait 5 seconds then retry


if __name__ == "__main__":
    asyncio.run(main())
