import atexit
from dataclasses import dataclass
from typing import Callable

import pika
import pika.adapters.blocking_connection
import pika.channel
from pika import spec


@dataclass
class RabbitConfig:
    user: str
    password: str
    host: str
    port: int


class PikaInitializer:
    def __init__(self, config: RabbitConfig, heartbeat: int) -> None:
        self.config = config
        self.credentials = pika.PlainCredentials(config.user, config.password)
        self.parameters = pika.ConnectionParameters(
            host=config.host,
            port=config.port,
            credentials=self.credentials,
            heartbeat=heartbeat,
        )
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()
        atexit.register(self._close_connection)

    def _close_connection(self) -> None:
        if self.channel and self.channel.is_open:
            self.channel.close()
        if self.connection and self.connection.is_open:
            self.connection.close()


@dataclass
class RabbitBinding:
    exchange: str
    queue: str
    routing_key: str


class RabbitMQClient:
    def __init__(
        self,
        channel: pika.adapters.blocking_connection.BlockingChannel,
        binding: RabbitBinding,
        declare: bool,
    ) -> None:
        self._channel = channel
        self._binding = binding
        if declare:
            self._channel.exchange_declare(binding.exchange, durable=True)
            self._channel.queue_declare(binding.queue, durable=True)
            self._channel.queue_bind(
                binding.queue, binding.exchange, binding.routing_key
            )

    def publish_message(self, message: str) -> None:
        self._channel.basic_publish(
            exchange=self._binding.exchange,
            routing_key=self._binding.routing_key,
            body=message,
        )

    def listen(self, callback: Callable | None, blocking: bool = True) -> None | str:
        if blocking:
            if callback is None:
                raise ValueError("Callback must be provided when blocking is True")
            self._channel.basic_consume(
                queue=self._binding.queue, on_message_callback=callback, auto_ack=True
            )
            self._channel.start_consuming()
            return None
        else:
            method_frame: spec.Basic.GetOk | None
            header_frame: spec.BasicProperties | None
            body: bytes | None
            decoded_body: str | None = None
            method_frame, header_frame, body = self._channel.basic_get(
                queue=self._binding.queue, auto_ack=True
            )
            if (
                method_frame is not None
                and header_frame is not None
                and body is not None
            ):
                decoded_body = body.decode("utf-8")
            return decoded_body


def setup_rabbitmq(
    user: str,
    password: str,
    host: str,
    port: int,
    heartbeat: int,
    topic: str,
    declare: bool,
) -> RabbitMQClient:
    rabbitmq_config = RabbitConfig(
        user,
        password,
        host,
        port,
    )
    init = PikaInitializer(rabbitmq_config, heartbeat)
    binding = RabbitBinding(exchange="fl", queue=topic, routing_key=topic)
    return RabbitMQClient(init.channel, binding, declare=declare)
