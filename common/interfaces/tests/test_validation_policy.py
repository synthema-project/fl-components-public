from interfaces.validation_policy import required_reports


def test_fidelity_is_opt_in(monkeypatch):
    monkeypatch.delenv('FIDELITY_ENABLED', raising=False)
    assert required_reports() == ['utility', 'privacy']
    monkeypatch.setenv('FIDELITY_ENABLED', 'true')
    assert required_reports() == ['utility', 'privacy', 'fidelity']
