from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_db
from app.main import app


def _mock_db():
    db = MagicMock()
    db.execute.return_value = MagicMock()
    yield db


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = _mock_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
