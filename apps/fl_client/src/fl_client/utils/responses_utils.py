from flwr.common import (
    DEFAULT_TTL,
    ConfigsRecord,
    Message,
    MetricsRecord,
    ParametersRecord,
    RecordSet,
)


def create_participate_response(msg: Message, participate: bool) -> Message:
    configsrecord = ConfigsRecord({"participate": participate})
    rs = RecordSet(configs_records={"config": configsrecord})
    return msg.create_reply(content=rs, ttl=DEFAULT_TTL)


def create_success_response(msg: Message, event: str) -> Message:
    configsrecord = ConfigsRecord({"success": True, "message": event})
    rs = RecordSet(configs_records={"config": configsrecord})
    return msg.create_reply(content=rs, ttl=DEFAULT_TTL)


def create_upload_model_response(msg: Message, model_id: str, run_id: int) -> Message:
    configsrecord = ConfigsRecord(
        {
            "success": True,
            "model_id": model_id,
            "run_id": run_id,
            "message": "model uploaded successfully",
        }
    )
    rs = RecordSet(configs_records={"config": configsrecord})
    return msg.create_reply(content=rs, ttl=DEFAULT_TTL)


def create_train_response(
    msg: Message, model_parameters: ParametersRecord, model_metrics: MetricsRecord
) -> Message:
    configsrecord = ConfigsRecord(
        {
            "success": True,
            "message": "model trained successfully",
        }
    )
    rs = RecordSet(
        parameters_records={"parameters": model_parameters},
        metrics_records={"metrics": model_metrics},
        configs_records={"config": configsrecord},
    )
    return msg.create_reply(content=rs, ttl=DEFAULT_TTL)


def create_parameters_response(
    msg: Message, model_parameters: ParametersRecord
) -> Message:
    configsrecord = ConfigsRecord(
        {
            "success": True,
            "message": "retrieved parameters",
        }
    )
    rs = RecordSet(
        parameters_records={"parameters": model_parameters},
        configs_records={"config": configsrecord},
    )
    return msg.create_reply(content=rs, ttl=DEFAULT_TTL)
