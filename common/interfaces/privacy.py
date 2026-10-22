"""Model-defined MSI privacy budgets and MLflow scalar projection."""
import math
from numbers import Real


def budget(value):
    if value is None or value == {}:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        value = dict(zip(('epsilon', 'delta'), value))
    if not isinstance(value, dict) or not {'epsilon', 'delta'} <= value.keys():
        raise ValueError('Invalid model privacy budget: expected epsilon and delta')
    epsilon, delta = value['epsilon'], value['delta']
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
           for x in (epsilon, delta)) or epsilon < 0 or not 0 <= delta < 1:
        raise ValueError('Invalid model privacy budget values')
    return {'epsilon': float(epsilon), 'delta': float(delta)}


def scalar_metrics(metrics):
    """Keep Flower records intact for aggregators, project numeric values for MLflow."""
    result = {key: float(value) for key, value in metrics.items()
              if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)}
    if 'privacy_budget' in metrics:
        reported = budget(metrics['privacy_budget'])
        if reported:
            result.update({f'privacy/{key}': value for key, value in reported.items()})
    return result
