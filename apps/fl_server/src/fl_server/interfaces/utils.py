from functools import wraps
import inspect
from typing import Any, Callable


class MutableBoolean:
    def __init__(self, value: bool) -> None:
        self._value = value

    def __bool__(self) -> bool:
        return self.value

    @property
    def value(self) -> bool:
        return self._value

    @value.setter
    def value(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("value must be a boolean")
        self._value = value


def ensure_bool(is_configured_var: MutableBoolean, value: bool = True) -> Callable:
    if not isinstance(is_configured_var, MutableBoolean):
        raise TypeError("is_configured_var must be an instance of MutableBoolean")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: list[Any], **kwargs: dict[str, Any]) -> Any:
            if bool(is_configured_var) is not value:
                raise RuntimeError(
                    f"The module {inspect.getmodule(func)} expects {value}, received {bool(is_configured_var)}."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
