import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from src.app.db.base import Base, get_db
from src.app.main import app

# Use an in-memory SQLite DB for tests
TEST_DATABASE_URL = "sqlite:///:memory:"


engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_env() -> None:
    """
    Global test env overrides. Adjust as needed (e.g. set LLM_PROVIDER, EMBEDDING_PROVIDER).
    """
    os.environ.setdefault("LLM_PROVIDER", "ollama")  # or "openai" if you test that
    os.environ.setdefault("EMBEDDING_PROVIDER", "huggingface")
    # Point VECTOR_DIR / FILE_DIR to temp locations in real tests if needed
    # os.environ.setdefault("VECTOR_DIR", "/tmp/test-vectorstore")


@pytest.fixture(scope="session")
def db_engine() -> Generator:
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """
    New database session per test.
    """
    connection = db_engine.connect()
    txn = connection.begin()
    session = TestingSessionLocal(bind=connection)

    try:
        yield session
    finally:
        session.close()
        txn.rollback()
        connection.close()


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """
    FastAPI TestClient with DB dependency overridden to the test session.
    """

    def override_get_db() -> Generator[Session, None, None]:
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
