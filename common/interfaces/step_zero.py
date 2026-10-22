"""MSI Step 0 statistics; only summaries cross the distributed boundary."""
import copy
import math
from model_store_interface.step_zero import PlainStepZeroStrategy, StepZeroMetrics
from model_store_interface.metadata.models.type_constants import NUMERIC_TYPE_NAMES


def variables(payload):
    return payload.get('metadata', payload).get('variables', [])


def numeric_variables(payload):
    return [v for v in variables(payload)
            if str(v.get('datatype', v.get('type', ''))).lower() in NUMERIC_TYPE_NAMES]


def compute(train, metadata):
    strategy = PlainStepZeroStrategy()
    return [strategy.compute_local_metrics(train[v['name']].copy(), v['name']).model_dump()
            for v in numeric_variables(metadata) if v['name'] in train.columns]


def aggregate(payloads):
    """Match Flower simulation: recompute numeric statistics using all replies."""
    schemas = [p['metadata'] for p in payloads]
    if all(schema is None for schema in schemas):
        return None  # Legacy models without feature metadata.
    if any(schema is None for schema in schemas):
        raise ValueError('Step 0 requires feature metadata on every client')
    signature = lambda schema: sorted((v['name'], str(v.get('datatype', v.get('type', ''))).lower())
                                      for v in variables(schema))
    if any(signature(schema) != signature(schemas[0]) for schema in schemas[1:]):
        raise ValueError('Step 0 clients have incompatible feature schemas')
    enriched = copy.deepcopy(schemas[0])
    numeric = numeric_variables(enriched)
    collected = {v['name']: [] for v in numeric}
    for payload in payloads:
        seen = set()
        for value in payload['metrics']:
            metric = StepZeroMetrics.model_validate(value)
            if metric.feature_name not in collected or metric.feature_name in seen:
                raise ValueError('Step 0 received unexpected or duplicate feature statistics')
            if (metric.count < 0 or metric.sum_sq_val < 0 or not all(math.isfinite(x)
                    for x in (metric.count, metric.sum_val, metric.sum_sq_val))):
                raise ValueError('Step 0 requires finite non-negative counts and squared sums')
            seen.add(metric.feature_name)
            collected[metric.feature_name].append(metric)
    strategy = PlainStepZeroStrategy()
    for variable in numeric:
        metrics = collected[variable['name']]
        if metrics:
            variable['mean'], variable['std'] = strategy.aggregate_global_stats(metrics)
    return enriched
