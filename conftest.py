# Root conftest.py to configure pytest behavior
#
# This file prevents pytest from trying to import the project's __init__.py
# which contains relative imports that don't work in test context.

import sys
import os
from pathlib import Path

# Add project root to path so tests can import nodes directly
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Store original __init__.py path
ROOT_INIT_PY = PROJECT_ROOT / "__init__.py"
BACKUP_INIT_PY = PROJECT_ROOT / "__init__.py.pytest_backup"


def pytest_configure(config):
    """
    Configure pytest by temporarily hiding the root __init__.py.

    The root __init__.py has relative imports that fail when pytest
    tries to import it as a package init. We rename it during test
    execution and restore it afterwards.
    """
    if ROOT_INIT_PY.exists() and not BACKUP_INIT_PY.exists():
        # Temporarily rename __init__.py to prevent pytest from importing it
        ROOT_INIT_PY.rename(BACKUP_INIT_PY)


def pytest_unconfigure(config):
    """
    Restore the root __init__.py after tests complete.
    """
    if BACKUP_INIT_PY.exists():
        # Restore original __init__.py
        BACKUP_INIT_PY.rename(ROOT_INIT_PY)
