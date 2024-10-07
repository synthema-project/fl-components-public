from threading import Thread, Event
from time import sleep
from typing import Any

from sqlmodel import Session

from restapi import config
from restapi.db import engine

from schemas.task import Task
from interfaces.rabbitmq_client import RabbitMQClient


class StatusUpdates(Thread):
    def __init__(self, event: Event, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.event = event

    def run(self) -> None:
        rabbitmq: RabbitMQClient = config.obj["rabbitmq_status"]
        while not self.event.is_set():
            body = rabbitmq.listen(None, blocking=False)
            if body is not None:
                message = body.split(" ")
                task_id = message[1]
                new_status = message[2]
                print(f"Received status update for task {task_id}: {new_status}")
                with Session(engine) as session:
                    task = session.get(Task, int(task_id))
                    if task is None:
                        raise ValueError(f"Task with id {task_id} not found")
                    task.change_status(new_status)
                    session.commit()
                    session.refresh(task)
            sleep(10)
