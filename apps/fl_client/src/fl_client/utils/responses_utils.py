from flwr.common import (
    DEFAULT_TTL,
    ConfigsRecord,
    Message,
    MetricsRecord,
    ParametersRecord,
    RecordSet,
)


def create_participate_response(msg: Message, participate: bool, node_name: str) -> Message:
    configsrecord = ConfigsRecord({"success": True, "participate": participate, "node_name": node_name})
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


def create_synthetic_data_response(
    msg: Message,
    object_uri: str,
    node_name: str,
    metadata_uri: str,
    evaluation_uri: str,
    utility_reporting_enabled: bool = False,
) -> Message:
    configsrecord = ConfigsRecord({
        "success": True,
        "object_uri": object_uri,
        "node_name": node_name,
        "metadata_uri": metadata_uri,
        "evaluation_uri": evaluation_uri,
        "utility_reporting_enabled": utility_reporting_enabled,
        "message": "synthetic data uploaded successfully",
    })
    return msg.create_reply(content=RecordSet(configs_records={"config": configsrecord}), ttl=DEFAULT_TTL)


def create_validation_response(msg: Message, result: dict) -> Message:
    import json
    return msg.create_reply(content=RecordSet(configs_records={
        "config": ConfigsRecord({"success": True, "message": "Validation reports uploaded"}),
        "validation": ConfigsRecord({
            "node_name": result["node_name"],
            "reports_json": json.dumps(result.get("reports", [])),
            "report_uri": result.get("report_uri", ""),
            "metrics_json": json.dumps(result.get("metrics", {})),
        }),
    }), ttl=DEFAULT_TTL)


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

def create_discovery_response(
    msg: Message, peers: dict[str, str]
) -> Message:
    configsrecord = ConfigsRecord(
        {
            "success": True,
            "message": "discovery complete",
            "peers": peers,
        }
    )
    rs = RecordSet(
        configs_records={"config": configsrecord},
    )
    return msg.create_reply(content=rs, ttl=DEFAULT_TTL)

def create_connect_peers_response(
    msg: Message, peers: list[str]
) -> Message:
    configsrecord = ConfigsRecord(
        {
            "success": True,
            "message": "connected to peers",
            "peers": peers
        }
    )
    rs = RecordSet(
        configs_records={"config": configsrecord},
    )
    return msg.create_reply(content=rs, ttl=DEFAULT_TTL)

def create_start_p2p_server_response(
    msg: Message, grpc_port: int
) -> Message:
    configsrecord = ConfigsRecord(
        {
            "success": True,
            "message": "started P2P server",
            "grpc_port": grpc_port
        }
    )
    rs = RecordSet(
        configs_records={"config": configsrecord},
    )
    return msg.create_reply(content=rs, ttl=DEFAULT_TTL)

def create_error_response(
    msg: Message, error_message: str
) -> Message:
    configsrecord = ConfigsRecord(
        {
            "success": False,
            "message": error_message,
        }
    )
    rs = RecordSet(configs_records={"config": configsrecord})
    return msg.create_reply(content=rs, ttl=DEFAULT_TTL)
