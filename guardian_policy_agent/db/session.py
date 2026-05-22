from __future__ import annotations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ..config import load_config

_engine = None
_Session = None

def init_engine(db_url: str, echo: bool = False):
    global _engine, _Session
    _engine = create_engine(db_url, echo=echo, future=True)
    _Session = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    return _engine

def get_engine():
    if _engine is None:
        cfg = load_config()
        init_engine(cfg.db_url, cfg.echo_sql)
    return _engine

def get_session():
    if _Session is None:
        get_engine()
    return _Session()
