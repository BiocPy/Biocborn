"""Dummy conftest.py for biocborn.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

import data.mock_sce as mocks
import pytest


@pytest.fixture
def mock_data():
    return mocks
