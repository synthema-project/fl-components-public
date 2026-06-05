"""
conftest.py — shared pytest configuration for restapi.

Adds the ``common/`` package directory to ``sys.path`` so that ``interfaces``,
``schemas``, and other common modules are importable without being installed
as editable packages.
This mirrors the runtime environment where PYTHONPATH would include ``common/``.
"""
import sys
from pathlib import Path

# common/ is three levels up relative to apps/restapi/:
# apps/restapi/ -> apps/ -> fl-components/ -> common/
_COMMON_DIR = Path(__file__).parent.parent.parent / "common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
