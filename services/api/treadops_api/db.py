"""SQLAlchemy engine/session. PostgreSQL is the business truth; every mutation runs in an explicit transaction."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, TypeVar

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .settings import get_settings

T = TypeVar("T")


class Base(DeclarativeBase):
    pass


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or get_settings().database_url, pool_pre_ping=True, pool_size=10, max_overflow=10,
                         future=True)


@lru_cache
def _factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False, future=True)


def new_session() -> Session:
    return _factory()()


@contextmanager
def session_scope() -> Iterator[Session]:
    """One transaction: commit on success, roll back on any exception."""
    s = new_session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: request-scoped session; handlers commit explicitly."""
    s = new_session()
    try:
        yield s
    finally:
        s.close()


def must_get(db: Session, model: type[T], ident: Any, **kw: Any) -> T:
    """`db.get` for a row that must exist (FK target or already read in this flow); raises if missing."""
    row = db.get(model, ident, **kw)
    if row is None:
        raise LookupError(f"{model.__name__} {ident} not found")
    return row
