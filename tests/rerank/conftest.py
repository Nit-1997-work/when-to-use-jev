from __future__ import annotations

import pytest

from jevbench.rerank.data import Search
from tests.rerank.helpers import make_search


@pytest.fixture
def search() -> Search:
    return make_search()
