import pytest
from interfaces import utils


def test_ensure_configured():
    is_configured = utils.MutableBoolean(False)

    @utils.ensure_bool(is_configured)
    def test_func():
        pass

    with pytest.raises(RuntimeError):
        test_func()

    is_configured.value = True

    try:
        test_func()
    except RuntimeError:
        pytest.fail("test_func raised RuntimeError unexpectedly!")

    @utils.ensure_bool(is_configured, False)
    def test_func2():
        pass

    with pytest.raises(RuntimeError):
        test_func2()

    is_configured.value = False

    try:
        test_func2()
    except RuntimeError:
        pytest.fail("test_func2 raised RuntimeError unexpectedly!")
