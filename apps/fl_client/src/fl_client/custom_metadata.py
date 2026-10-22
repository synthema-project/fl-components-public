"""Optional MSI custom metadata hooks, run before learner preparation."""
from flwr.common import ConfigsRecord


def compute(global_vars):
    learner = global_vars['local_learner']
    hook = getattr(learner, 'compute_custom_metadata', None)
    if not callable(hook):
        return None
    result = hook(global_vars['data_partitions']['train'].copy(), global_vars.get('model_meta'))
    return None if result is None else ConfigsRecord(dict(result))


def apply(metadata, global_vars):
    hook = getattr(global_vars['local_learner'], 'set_custom_metadata', None)
    if callable(hook):
        hook(metadata)
