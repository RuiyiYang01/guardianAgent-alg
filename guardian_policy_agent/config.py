from __future__ import annotations
import os, yaml
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_DB_URL = os.getenv("DB_URL", "sqlite:///guardian_policy_agent.db")

@dataclass
class AppConfig:
    db_url: str
    echo_sql: bool = False

def load_config(path: str | None = None) -> AppConfig:
    db_url = DEFAULT_DB_URL
    echo_sql = False
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        db_url = y.get("db_url", db_url)
        echo_sql = bool(y.get("echo_sql", echo_sql))
    else:
        # Allow pure ENV usage
        db_url = os.getenv("DB_URL", db_url)
        echo_sql = os.getenv("ECHO_SQL", "false").lower() in {"1","true","yes","on"}
    return AppConfig(db_url=db_url, echo_sql=echo_sql)
