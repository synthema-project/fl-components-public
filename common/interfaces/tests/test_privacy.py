import pytest
from flwr.common import MetricsRecord
from interfaces.privacy import budget, scalar_metrics


@pytest.mark.parametrize('value', [[-1, 1e-5], [1, 1], [float('nan'), 1e-5], {}, None])
def test_invalid_or_missing_budgets_are_not_fabricated(value):
    if value is None or value == {}:
        assert budget(value) is None
    else:
        with pytest.raises(ValueError):
            budget(value)


def test_scalar_projection_does_not_mutate_model_metrics():
    metrics = MetricsRecord({'privacy_budget': [1., 1e-5], 'curve': [1., 2.], 'loss': .5})
    assert scalar_metrics(metrics) == {'privacy/epsilon': 1., 'privacy/delta': 1e-5, 'loss': .5}
    assert metrics['curve'] == [1., 2.]
    assert metrics['privacy_budget'] == [1., 1e-5]
