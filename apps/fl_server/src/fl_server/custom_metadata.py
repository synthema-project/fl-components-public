"""MSI Step 0.5: aggregate model-defined metadata before preparing clients."""
from flwr.common import ConfigsRecord


def exchange(global_vars, collect, broadcast):
    aggregate = getattr(global_vars['aggregator'], 'aggregate_custom_metadata', None)
    if not callable(aggregate):
        return False
    local = []
    for value in collect():
        if value is not None:
            local.append(ConfigsRecord(dict(value)))
    if not local:
        return False
    metadata = aggregate(local)
    if metadata is None:
        return False
    metadata = ConfigsRecord(dict(metadata))
    # Wait for all clients to acknowledge application before prepare_data.
    broadcast(metadata)
    model = global_vars['model'].unwrap_python_model()
    hook = getattr(model, 'set_custom_metadata', None)
    if callable(hook):
        hook(metadata)
    return True
