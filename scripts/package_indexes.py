"""Validate the shared private index location before passing them as Docker build arguments."""
import os
from urllib.parse import urlsplit

DEFAULT_INDEX = 'https://pypi.synthema.rid-intrasoft.eu/simple/'


def package_indexes(env):
    result = {}
    for name in ('SYNTHEMA_PYPI_URL',):
        value = env.get(name, DEFAULT_INDEX).strip()
        try:
            parsed = urlsplit(value)
            valid = (parsed.scheme == 'https' and parsed.hostname and
                     parsed.username is None and parsed.password is None and
                     not parsed.query and not parsed.fragment and
                     parsed.path.rstrip('/').endswith('/simple') and
                     not any(c.isspace() for c in value))
            parsed.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f'{name} must be an HTTPS PyPI /simple/ index URL without credentials, query or fragment')
        result[name] = value
    return result


if __name__ == '__main__':
    package_indexes(os.environ)
