"""
This file is the database engine + session factory for SQLAlchemy.

1.Creates SQLAlchemy engine (connection to Postgres)
2.Creates a session factory (per-request database sessions)
3.Defines Base (ORM model base class)
4.Provides get_db() dependency for FastAPI routes
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ..core.settings import DATABASE_URL

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

